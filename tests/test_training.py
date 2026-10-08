"""Validate the trainer's self-test, the server additions (tolerance ring, crash-safe checkpoints,
resumed epoch logs, saved overlays, initial weights) and the Swin encoder option."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from filmunet.evaluation import prediction_writer, spread_over_sources
    from filmunet.losses import FocalCrackLoss, HairlineCrackLoss
    from filmunet.models import HairlineFiLMSwinUNet, HairlineFiLMUNet, load_model_from_checkpoint
    from filmunet.self_test import run_self_tests
    from filmunet.training import initial_weights, open_epoch_log, parameter_groups, save_checkpoint
except ImportError:  # the workflow tests still run without PyTorch
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.zeros(1, 1, 32, 32)
        self.target[0, 0, 16, 4:28] = 1  # a one-pixel horizontal crack on row 16
        self.valid = torch.ones_like(self.target)

    def gradient(self, tolerance, loss=None):
        logits = torch.randn(1, 1, 32, 32, requires_grad=True)
        (loss or HairlineCrackLoss)(label_tolerance=tolerance)(logits, self.target, self.valid).backward()
        return logits.grad[0, 0]

    def test_built_in_self_test_passes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_self_tests()  # raises SystemExit with the failed checks
        self.assertIn('All self-tests passed.', output.getvalue())

    def test_strict_loss_supervises_pixels_beside_the_label(self):
        self.assertTrue((self.gradient(0)[14:19, 4:28] != 0).all())

    def test_tolerance_ring_is_left_out_of_the_loss(self):
        for loss in (HairlineCrackLoss, FocalCrackLoss):
            gradient = self.gradient(2, loss)
            self.assertTrue((gradient[[14, 15, 17, 18], 4:28] == 0).all())  # within 2 px of the label
            self.assertTrue((gradient[16, 4:28] != 0).all())                # the label itself
            self.assertTrue((gradient[[13, 19], 4:28] != 0).all())          # 3 px away: background again

    def test_checkpoint_is_replaced_whole(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            save_checkpoint({'epoch': 1}, path)
            save_checkpoint({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 2)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ['last.pt'])

    def test_resumed_log_drops_the_repeated_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training_history.csv'
            # Killed after logging epoch 3 but before saving last.pt, so epoch 3 runs again.
            path.write_text('epoch,loss\n1,0.9\n2,0.8\n3,0.7\n', encoding='utf-8')
            with open_epoch_log(path, ['epoch', 'loss'], 3) as handle:
                handle.write('3,0.6\n')
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss', '1,0.9', '2,0.8', '3,0.6'])
            with open_epoch_log(path, ['epoch', 'loss'], 1):  # a fresh run starts the file again
                pass
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss'])

    def test_saved_overlays_cover_every_source(self):
        domain_ids = [0] * 300 + [1] * 50 + [2] * 3  # one large source first, as in the size-grouped loader
        chosen = spread_over_sources(domain_ids, 40)
        per_source = [sum(domain_ids[index] == domain for index in chosen) for domain in range(3)]
        self.assertEqual(per_source, [19, 18, 3])  # the small source is saved whole, the rest shared
        self.assertGreater(max(index for index in chosen if domain_ids[index] == 0), 250)  # not just the first files
        self.assertEqual(spread_over_sources(domain_ids, 10**6), set(range(len(domain_ids))))  # a large limit saves all

    def test_overlays_are_written_for_the_selected_images_only(self):
        class Tiles:  # the parts of CrackSegmentationDataset the writer reads
            domain_ids = [0, 0, 0, 1]
            shapes = [(32, 32)] * 4
            pairs = [(Path(f'source{domain}_{index}.png'), None) for index, domain in enumerate(domain_ids)]

        with tempfile.TemporaryDirectory() as folder:
            write = prediction_writer(Tiles, Path(folder), 2, 0.5)
            write(torch.rand(4, 3, 32, 32), torch.rand(4, 1, 32, 32), self.target.expand(4, -1, -1, -1),
                  torch.arange(4))
            names = sorted(item.name for item in Path(folder).iterdir())
            self.assertEqual(names, ['source0_0_overlay.png', 'source0_0_probability.png',
                                     'source1_3_overlay.png', 'source1_3_probability.png'])

    def test_initial_weights_without_film_start_the_film_model_as_that_model(self):
        kwargs = dict(encoder_name='resnet18', decoder_channels=[256, 128, 64, 48, 32], stem_channels=32,
                      line_prior=True, deep_supervision=True, film_skips=False, film_hidden=64,
                      gsd_dropout=0.15, gsd_log_mean=0.2, gsd_log_std=0.5)
        plain = HairlineFiLMUNet(pretrained=False, use_film=False, **kwargs).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint({'model': plain.state_dict(),
                             'architecture': {'name': 'hairline_film_unet', 'kwargs': {**kwargs, 'use_film': False}}},
                            path)
            wanted = {**kwargs, 'use_film': True, 'gsd_log_mean': -0.4}
            initial = initial_weights(path, wanted)
            self.assertEqual(wanted['gsd_log_mean'], -0.4)  # no FiLM in the weights: this run's normalisation
            model = HairlineFiLMUNet(pretrained=False, **wanted).eval()
            keys = model.load_state_dict(initial['model'], strict=False)
            self.assertFalse(keys.unexpected_keys)
            self.assertTrue(keys.missing_keys and all(key.startswith('film.') for key in keys.missing_keys))
            images = torch.rand(2, 3, 64, 96)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(images, torch.tensor([0.65, 2.5])), plain(images)))
            with self.assertRaises(SystemExit):  # another encoder is another network
                initial_weights(path, {**wanted, 'encoder_name': 'resnet34'})
            # Weights with FiLM keep the normalisation they were trained with, and need FiLM again.
            save_checkpoint({'model': model.state_dict(),
                             'architecture': {'name': 'hairline_film_unet', 'kwargs': wanted}}, path)
            again = {**wanted, 'gsd_log_mean': 1.0}
            initial_weights(path, again)
            self.assertEqual(again['gsd_log_mean'], -0.4)
            with self.assertRaises(SystemExit):
                initial_weights(path, {**wanted, 'use_film': False})
            with self.assertRaises(SystemExit):
                initial_weights(path, {**wanted, 'film_skips': True})


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class SwinEncoderTests(unittest.TestCase):
    """--encoder swin_t: the checks the self-test makes on the ResNet network."""

    kwargs = dict(encoder_name='swin_t', pretrained=False, gsd_log_mean=0.3, gsd_log_std=0.5)

    @staticmethod
    def build(**changes):
        torch.manual_seed(0)
        model = HairlineFiLMSwinUNet(**{**SwinEncoderTests.kwargs, **changes})
        for module in model.modules():
            if type(module).__name__ == 'StochasticDepth':
                module.p = 0.0  # so that two training-mode passes can be compared
        return model

    def test_zero_initialised_film_is_the_plain_network(self):
        images, gsd = torch.rand(2, 3, 96, 128), torch.tensor([0.65, float('nan')])
        for skips in (False, True):
            film, plain = self.build(film_skips=skips), self.build(use_film=False)
            keys = plain.load_state_dict(film.state_dict(), strict=False)
            self.assertFalse(keys.missing_keys)
            self.assertTrue(all(key.startswith('film.') for key in keys.unexpected_keys))
            self.assertEqual(len(film.film.heads), 10 if skips else 5)
            with torch.no_grad():
                self.assertTrue(torch.equal(film.eval()(images, gsd), plain.eval()(images, gsd)))
                film_logits, film_auxiliary = film.train()(images, gsd)
                plain_logits, plain_auxiliary = plain.train()(images, gsd)
            self.assertEqual(film_logits.shape, (2, 1, 96, 128))
            self.assertTrue(torch.equal(film_logits, plain_logits))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(film_auxiliary, plain_auxiliary)))

    def test_trained_film_reacts_to_the_gsd_and_handles_unknown(self):
        model = self.build(film_skips=True).eval()
        images, nan = torch.rand(2, 3, 96, 128), torch.full((2,), float('nan'))
        with torch.no_grad():
            for head in model.film.heads.values():
                head.weight.normal_(0, 0.1)
            known, other = model(images, torch.tensor([0.65, 0.65])), model(images, torch.tensor([2.5, 2.5]))
            unknown = model(images, nan)
            self.assertFalse(torch.allclose(known, other))
            self.assertTrue(bool(torch.isfinite(unknown).all()) and torch.equal(unknown, model(images)))
            mixed = model(images, torch.tensor([0.65, float('nan')]))
            self.assertTrue(torch.allclose(mixed[0], known[0], atol=1e-4))
            self.assertTrue(torch.allclose(mixed[1], unknown[1], atol=1e-4))
            model.train()
            model.gsd_dropout = 1.0
            dropped = model(images, torch.tensor([0.65, 0.65]))[0]
            model.gsd_dropout = 0.0
            self.assertTrue(torch.allclose(dropped, model(images, nan)[0], atol=1e-4))

    def test_gradients_reach_film_and_its_parameters_are_not_decayed(self):
        model = self.build().train()
        images, masks = torch.rand(2, 3, 64, 64), (torch.rand(2, 1, 64, 64) > 0.9).float()
        loss = HairlineCrackLoss()(model(images, torch.tensor([0.65, float('nan')])), masks, torch.ones_like(masks))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(model.film.null_embedding.grad).all()))
        self.assertIsNotNone(model.film.embed.weight.grad)
        named = dict(model.named_parameters())
        group_of = {id(parameter): group for group in parameter_groups(model, 2e-4, 0.3, 0.01)
                    for parameter in group['params']}
        for name, parameter in named.items():
            group = group_of[id(parameter)]
            if name.startswith('film.') or 'relative_position_bias_table' in name:
                self.assertEqual(group['weight_decay'], 0.0, name)
            self.assertAlmostEqual(group['lr'], 2e-4 * (0.3 if name.startswith('encoder.') else 1.0), msg=name)
        self.assertEqual(group_of[id(named['up3.fuse.weight'])]['weight_decay'], 0.01)

    def test_checkpoint_rebuilds_the_swin_network(self):
        model = self.build(film_skips=True).eval()
        architecture = {'name': 'hairline_film_swin_unet', 'kwargs': {
            'encoder_name': 'swin_t', 'decoder_depth': 2, 'head_channels': [48, 32], 'stem_channels': 32,
            'line_prior': True, 'deep_supervision': True, 'use_film': True, 'film_skips': True, 'film_hidden': 64,
            'gsd_dropout': 0.15, 'gsd_log_mean': 0.3, 'gsd_log_std': 0.5}}
        images, gsd = torch.rand(1, 3, 64, 64), torch.tensor([1.9])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint({'model': model.state_dict(), 'architecture': architecture}, path)
            rebuilt, _ = load_model_from_checkpoint(path)
            with torch.no_grad():
                self.assertTrue(torch.equal(rebuilt(images, gsd), model(images, gsd)))


if __name__ == '__main__':
    unittest.main()
