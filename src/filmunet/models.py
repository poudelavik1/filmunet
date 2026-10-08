'Models for hairline crack segmentation.'

from __future__ import annotations
from .common import ENCODER_CHANNELS, F, copy, math, nn, torch

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(ConvBNReLU(input_channels, output_channels), ConvBNReLU(output_channels, output_channels))


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)
        self.attention = SCSE(output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.attention(self.conv(torch.cat((x, skip), 1)))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. Both are needed: in EH104
    many cracks appear lighter than the concrete (deposits in the crack), not
    darker. The module has no parameters, so it is a thin-line prior at no data cost.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)


class FiLMGenerator(nn.Module):
    """Turns a tile's GSD into a per-channel scale and shift for every conditioned feature map.

    normalized log(GSD) -> Linear(1, hidden) -> SiLU -> Linear(hidden, hidden) -> SiLU, then one
    Linear(hidden, 2C) per layer gives (d_gamma, beta) for F' = (1 + d_gamma) * F + beta.
    """

    def __init__(self, layer_channels: dict[str, int], hidden: int = 64):
        super().__init__()
        self.embed = nn.Linear(1, hidden)
        self.trunk = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 2 * channels)
                                    for name, channels in layer_channels.items()})
        # Zero heads give d_gamma = 0 and beta = 0, so F' = F: the network starts as the plain
        # U-Net with its pretrained encoder undisturbed, and uses the GSD only as far as the
        # gradient finds it useful.
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Stands in for embed(gsd) when the GSD is unknown. It starts as the embedding of the mean
        # training GSD (normalized input 0, so embed gives its bias): the best guess for a tile
        # nothing is known about.
        self.null_embedding = nn.Parameter(self.embed.bias.detach().clone())

    def forward(self, normalized_log_gsd: torch.Tensor, known: torch.Tensor) -> dict:
        embedding = self.embed(normalized_log_gsd[:, None])
        embedding = torch.where(known[:, None], embedding, self.null_embedding.to(embedding.dtype))
        hidden = self.trunk(embedding)
        return {name: head(hidden).view(len(hidden), 2, -1, 1, 1).unbind(1) for name, head in self.heads.items()}


class HairlineFiLMUNet(nn.Module):
    """ResNet-encoder U-Net with a full-resolution skip for 1-4 px cracks, conditioned on the GSD.

    Input: RGB in [0, 1] and `gsd`, a float tensor (N,) in mm/px with NaN (or None) for unknown.
    Eval mode returns logits (N, 1, H, W). Train mode with deep supervision returns
    (logits, [auxiliary logits]).

    ``use_film=False`` gives the same network without the conditioning (it ignores `gsd`): the
    baseline of the FiLM ablation.
    """

    def __init__(
        self, encoder_name: str = "resnet34", pretrained: bool = True,
        decoder_channels=(256, 128, 64, 48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
        use_film: bool = True, film_skips: bool = False, film_hidden: int = 64, gsd_dropout: float = 0.15,
        gsd_log_mean: float = 0.0, gsd_log_std: float = 1.0,
    ):
        super().__init__()
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("The encoder requires torchvision: python -m pip install torchvision") from exc
        try:
            self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load ImageNet {encoder_name} weights. In Kaggle, enable Internet "
                    "for the first run, or pass --no-pretrained."
                ) from exc
            raise
        self.encoder.fc = nn.Identity()
        c0, c1, c2, c3, c4 = ENCODER_CHANNELS[encoder_name]
        d4, d3, d2, d1, d0 = decoder_channels
        self.thin_lines = ThinLineResponse() if line_prior else None
        stem_inputs = 3 + (self.thin_lines.output_channels if line_prior else 0)
        self.full_resolution = DoubleConv(stem_inputs, stem_channels)
        self.dec4 = DecoderBlock(c4, c3, d4)              # 1/16
        self.dec3 = DecoderBlock(d4, c2, d3)              # 1/8
        self.dec2 = DecoderBlock(d3, c1, d2)              # 1/4
        self.dec1 = DecoderBlock(d2, c0, d1)              # 1/2
        self.dec0 = DecoderBlock(d1, stem_channels, d0)   # 1/1
        self.head = nn.Conv2d(d0, 1, 1)
        self.deep_supervision = deep_supervision
        # Auxiliary heads on the decoder outputs at 1/8 and 1/4 scale.
        self.auxiliary_heads = nn.ModuleList([nn.Conv2d(d3, 1, 1), nn.Conv2d(d2, 1, 1)]) if deep_supervision else None
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        # FiLM: one (d_gamma, beta) pair per decoder block output, and per skip with film_skips.
        self.film = None
        if use_film:
            layers = {"dec4": d4, "dec3": d3, "dec2": d2, "dec1": d1, "dec0": d0}
            if film_skips:
                layers.update({"skip4": c3, "skip3": c2, "skip2": c1, "skip1": c0, "skip0": stem_channels})
            self.film = FiLMGenerator(layers, film_hidden)
        self.gsd_dropout = gsd_dropout
        # Plain numbers, not buffers: they are fixed statistics of the training set, and the EMA
        # would otherwise average them like weights.
        self.gsd_log_mean, self.gsd_log_std = float(gsd_log_mean), float(gsd_log_std)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def film_parameters(self, gsd, batch: int, device) -> dict:
        """{layer: (d_gamma, beta)} for the batch; empty without FiLM."""
        if self.film is None:
            return {}
        if gsd is None:
            gsd = float("nan")
        gsd = torch.as_tensor(gsd, dtype=torch.float32, device=device).reshape(-1)
        if gsd.numel() == 1:
            gsd = gsd.expand(batch)
        known = torch.isfinite(gsd) & (gsd > 0)
        if self.training and self.gsd_dropout > 0:
            # GSD dropout: some known tiles are trained as "unknown", so the null embedding is
            # trained on every source and the model does not come to depend on being told the GSD.
            known = known & (torch.rand(batch, device=device) >= self.gsd_dropout)
        # log(GSD), standardised with the training-set statistics. Unknown entries get a dummy 1
        # so the logarithm stays finite; their embedding is replaced by the null embedding.
        normalized = (torch.log(torch.where(known, gsd, torch.ones_like(gsd))) - self.gsd_log_mean) / self.gsd_log_std
        return self.film(normalized, known)

    @staticmethod
    def modulate(features: torch.Tensor, film: dict, layer: str) -> torch.Tensor:
        """F' = (1 + d_gamma) * F + beta; the identity when `layer` is not conditioned."""
        if layer not in film:
            return features
        delta_gamma, beta = film[layer]
        return features * (1 + delta_gamma) + beta

    def forward(self, x, gsd=None):
        size = x.shape[-2:]
        film = self.film_parameters(gsd, x.size(0), x.device)
        modulate = self.modulate
        normalized = (x - self.image_mean) / self.image_std
        stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
        full = self.full_resolution(stem_input)
        encoder = self.encoder
        e0 = encoder.relu(encoder.bn1(encoder.conv1(normalized)))   # 1/2
        e1 = encoder.layer1(encoder.maxpool(e0))                     # 1/4
        e2 = encoder.layer2(e1)                                      # 1/8
        e3 = encoder.layer3(e2)                                      # 1/16
        e4 = encoder.layer4(e3)                                      # 1/32
        d4 = modulate(self.dec4(e4, modulate(e3, film, "skip4")), film, "dec4")      # 1/16
        d3 = modulate(self.dec3(d4, modulate(e2, film, "skip3")), film, "dec3")      # 1/8
        d2 = modulate(self.dec2(d3, modulate(e1, film, "skip2")), film, "dec2")      # 1/4
        d1 = modulate(self.dec1(d2, modulate(e0, film, "skip1")), film, "dec1")      # 1/2
        d0 = modulate(self.dec0(d1, modulate(full, film, "skip0")), film, "dec0")    # 1/1
        logits = self.head(d0)
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(features), size=size, mode="bilinear", align_corners=False)
                for head, features in zip(self.auxiliary_heads, (d3, d2))
            ]
            return logits, auxiliary
        return logits


class PatchExpand(nn.Module):
    """Swin-Unet patch expanding, the inverse of patch merging.

    A linear layer widens the channels, then every token is rearranged into
    scale x scale tokens. Tokens are channels-last: (N, H, W, C).
    """

    def __init__(self, input_channels: int, output_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.expand = nn.Linear(input_channels, output_channels * scale * scale, bias=False)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        batch, height, width, _ = x.shape
        scale = self.scale
        x = self.expand(x).reshape(batch, height, width, scale, scale, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * scale, width * scale, -1)
        return self.norm(x)


class SwinDecoderStage(nn.Module):
    """Swin-Unet up-stage: patch expanding, skip concatenation, linear fusion, Swin blocks."""

    def __init__(self, input_channels: int, blocks: nn.Sequential):
        super().__init__()
        output_channels = input_channels // 2
        self.expand = PatchExpand(input_channels, output_channels)
        self.fuse = nn.Linear(2 * output_channels, output_channels)
        self.blocks = blocks

    def forward(self, x, skip):
        # Patch merging pads odd sizes, so the expanded map can be one token larger than the skip.
        x = self.expand(x)[:, :skip.shape[1], :skip.shape[2]]
        return self.blocks(self.fuse(torch.cat((x, skip), -1)))


class HairlineFiLMSwinUNet(nn.Module):
    """The Swin-Unet of the Swin U-Net project (``--encoder swin_*``), conditioned on the GSD.

    HairlineSwinUNet of that project (Swin-Unet, Cao et al., 2021, with a full-resolution head)
    with the FiLM conditioning of HairlineFiLMUNet: the same inputs, outputs and options.

    * Encoder: an ImageNet-pretrained torchvision Swin Transformer giving token maps at 1/4,
      1/8, 1/16 and 1/32 scale.
    * Decoder: three Swin up-stages at 1/16, 1/8 and 1/4 scale (patch expanding, skip, linear
      fusion and two Swin blocks that start from the encoder stage they mirror), then the two
      DecoderBlocks of HairlineFiLMUNet at 1/2 and 1/1 scale, fed by the thin-line stem.
    * FiLM layers are named by scale as in HairlineFiLMUNet: dec4 ... dec0 are the outputs of the
      five decoder stages, skip4 ... skip0 the skips that enter them.

    The parameter names are HairlineSwinUNet's, so a best.pt of the Swin U-Net project loads
    with only the ``film.*`` entries missing (``--init-weights``).
    """

    def __init__(
        self, encoder_name: str = "swin_t", pretrained: bool = True, decoder_depth: int = 2,
        head_channels=(48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
        use_film: bool = True, film_skips: bool = False, film_hidden: int = 64, gsd_dropout: float = 0.15,
        gsd_log_mean: float = 0.0, gsd_log_std: float = 1.0,
    ):
        super().__init__()
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("The encoder requires torchvision: python -m pip install torchvision") from exc
        try:
            self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load ImageNet {encoder_name} weights. In Kaggle, enable Internet "
                    "for the first run, or pass --no-pretrained."
                ) from exc
            raise
        self.encoder.head = nn.Identity()
        # encoder.features = [patch embedding, stage 1, merge, stage 2, merge, stage 3, merge, stage 4]
        stages = self.encoder.features
        width = stages[0][0].out_channels  # C; the stages carry C, 2C, 4C, 8C channels

        def mirror(stage):
            # As in the paper, a decoder stage starts from the weights of the encoder stage it mirrors.
            return nn.Sequential(*copy.deepcopy(list(stage)[:decoder_depth]))

        self.up3 = SwinDecoderStage(8 * width, mirror(stages[5]))   # 1/16, 4C
        self.up2 = SwinDecoderStage(4 * width, mirror(stages[3]))   # 1/8,  2C
        self.up1 = SwinDecoderStage(2 * width, mirror(stages[1]))   # 1/4,  C
        self.decoder_norm = nn.LayerNorm(width)
        d1, d0 = head_channels
        self.thin_lines = ThinLineResponse() if line_prior else None
        stem_inputs = 3 + (self.thin_lines.output_channels if line_prior else 0)
        self.full_resolution = DoubleConv(stem_inputs, stem_channels)                                        # 1/1
        self.half_resolution = nn.Sequential(nn.MaxPool2d(2), DoubleConv(stem_channels, 2 * stem_channels))  # 1/2
        self.dec1 = DecoderBlock(width, 2 * stem_channels, d1)   # 1/2
        self.dec0 = DecoderBlock(d1, stem_channels, d0)          # 1/1
        self.head = nn.Conv2d(d0, 1, 1)
        self.deep_supervision = deep_supervision
        # Auxiliary heads on the Swin decoder tokens at 1/8 and 1/4 scale.
        self.auxiliary_heads = (
            nn.ModuleList([nn.Linear(2 * width, 1), nn.Linear(width, 1)]) if deep_supervision else None
        )
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        # FiLM: one (d_gamma, beta) pair per decoder stage output, and per skip with film_skips.
        self.film = None
        if use_film:
            layers = {"dec4": 4 * width, "dec3": 2 * width, "dec2": width, "dec1": d1, "dec0": d0}
            if film_skips:
                layers.update({"skip4": 4 * width, "skip3": 2 * width, "skip2": width,
                               "skip1": 2 * stem_channels, "skip0": stem_channels})
            self.film = FiLMGenerator(layers, film_hidden)
        self.gsd_dropout = gsd_dropout
        # Plain numbers, not buffers, as in HairlineFiLMUNet.
        self.gsd_log_mean, self.gsd_log_std = float(gsd_log_mean), float(gsd_log_std)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    # The GSD embedding, its dropout and the modulation are HairlineFiLMUNet's.
    film_parameters = HairlineFiLMUNet.film_parameters
    modulate = staticmethod(HairlineFiLMUNet.modulate)

    @staticmethod
    def modulate_tokens(tokens: torch.Tensor, film: dict, layer: str) -> torch.Tensor:
        """modulate() for Swin tokens, which are channels-last: (N, H, W, C)."""
        if layer not in film:
            return tokens
        delta_gamma, beta = film[layer]
        return tokens * (1 + delta_gamma.permute(0, 2, 3, 1)) + beta.permute(0, 2, 3, 1)

    def forward(self, x, gsd=None):
        size = x.shape[-2:]
        film = self.film_parameters(gsd, x.size(0), x.device)
        modulate, modulate_tokens = self.modulate, self.modulate_tokens
        normalized = (x - self.image_mean) / self.image_std
        stages = self.encoder.features
        e1 = stages[1](stages[0](normalized))                    # 1/4, tokens (N, H/4, W/4, C)
        e2 = stages[3](stages[2](e1))                            # 1/8
        e3 = stages[5](stages[4](e2))                            # 1/16
        e4 = self.encoder.norm(stages[7](stages[6](e3)))         # 1/32, bottleneck
        d4 = modulate_tokens(self.up3(e4, modulate_tokens(e3, film, "skip4")), film, "dec4")     # 1/16
        d3 = modulate_tokens(self.up2(d4, modulate_tokens(e2, film, "skip3")), film, "dec3")     # 1/8
        # Modulated after the norm, which would otherwise rescale what FiLM has just set.
        d2 = modulate_tokens(self.decoder_norm(self.up1(d3, modulate_tokens(e1, film, "skip2"))), film, "dec2")
        stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
        full = self.full_resolution(stem_input)
        half = self.half_resolution(full)
        d1 = modulate(self.dec1(d2.permute(0, 3, 1, 2), modulate(half, film, "skip1")), film, "dec1")   # 1/2
        d0 = modulate(self.dec0(d1, modulate(full, film, "skip0")), film, "dec0")                        # 1/1
        logits = self.head(d0)
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(tokens).permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False)
                for head, tokens in zip(self.auxiliary_heads, (d3, d2))
            ]
            return logits, auxiliary
        return logits


# Checkpoint "architecture" names and the networks they rebuild.
ARCHITECTURES = {"hairline_film_unet": HairlineFiLMUNet, "hairline_film_swin_unet": HairlineFiLMSwinUNet}


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild the network of a best.pt for inference scripts.

    The architecture kwargs in the checkpoint carry the FiLM flag and the GSD normalisation, so the
    model is called as ``model(images, gsd)`` with the GSD in mm/px (NaN or None when unknown).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = ARCHITECTURES[architecture["name"]](pretrained=False, **architecture["kwargs"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class ModelEMA:
    """Exponential moving average of weights (and BN statistics)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


__all__ = [
    'ConvBNReLU', 'DoubleConv', 'SCSE', 'DecoderBlock', 'ThinLineResponse', 'FiLMGenerator', 'HairlineFiLMUNet',
    'PatchExpand', 'SwinDecoderStage', 'HairlineFiLMSwinUNet', 'ARCHITECTURES', 'load_model_from_checkpoint',
    'ModelEMA',
]
