"""
Vision Encoders for EmberVLM

Supports multiple vision backbones:
- RepViT: Lightweight CNN-based encoder
- MobileViT-XS: Apple's mobile vision transformer
- DINOv2-Small: Self-supervised vision transformer (RECOMMENDED)

DINOv2 provides superior semantic features for vision-language tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from timm.models.layers import SqueezeExcite
    from timm.models.vision_transformer import trunc_normal_
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Vision backbone type constants
VISION_BACKBONE_REPVIT = "repvit"
VISION_BACKBONE_MOBILEVIT_XS = "mobilevit_xs"
VISION_BACKBONE_DINOV2_SMALL = "dinov2_small"


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure channel count is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    """Convolution with Batch Normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ):
        super().__init__()
        self.add_module(
            'c',
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias=False
            )
        )
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse conv and bn for inference."""
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5

        m = nn.Conv2d(
            w.size(1) * c.groups, w.size(0), w.shape[2:],
            stride=c.stride, padding=c.padding,
            dilation=c.dilation, groups=c.groups,
            device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """Residual connection with optional dropout."""

    def __init__(self, module: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = module
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            mask = mask.ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x) * mask
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(nn.Module):
    """RepVGG-style depthwise convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d_BN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.dim = channels
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias if conv1.bias is not None else torch.zeros_like(
            conv_b)

        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])
        identity = F.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1],
                       1, 1, device=conv1_w.device),
            [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * \
            bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    """RepViT building block."""

    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):
    """RepViT backbone network."""

    def __init__(
        self,
        cfgs: list,
        num_classes: int = 1000,
        distillation: bool = False,
    ):
        super().__init__()
        self.cfgs = cfgs

        # Build first layer (patch embedding)
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        )
        layers = [patch_embed]

        # Build RepViT blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size,
                          output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.num_features = output_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.features:
            x = f(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps without classification head."""
        return self.forward(x)


def repvit_xxs(pretrained: bool = False) -> RepViT:
    """
    Construct RepViT-XXS model (extra extra small variant).
    This is optimized for edge deployment.
    """
    # Configuration for XXS variant
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 64, 0, 0, 2],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 1, 2],
        [3, 2, 128, 1, 1, 1],
        [3, 2, 128, 0, 1, 1],
        [3, 2, 128, 1, 1, 1],
        [3, 2, 128, 0, 1, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
    ]
    model = RepViT(cfgs, num_classes=0, distillation=False)

    if pretrained:
        # Load pretrained weights from timm
        try:
            import timm
            # Use timm's repvit_m0_9 as a lightweight pretrained base
            # This is the closest match to xxs size while being available
            pretrained_model = timm.create_model(
                'repvit_m0_9.dist_450e_in1k', pretrained=True)

            # Extract compatible weights (feature extractor part)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()

            # Filter out incompatible keys (classifier head, size mismatches)
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v

            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            print(
                f"Loaded {len(compatible_dict)}/{len(model_dict)} weights from timm/repvit_m0_9.dist_450e_in1k")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    return model


class RepViTEncoder(nn.Module):
    """
    RepViT Vision Encoder wrapper for EmberVLM.

    Handles image preprocessing, feature extraction, and adaptive pooling
    to produce a fixed number of visual tokens.
    """

    def __init__(
        self,
        model_name: str = "repvit_xxs",
        pretrained: bool = True,
        freeze: bool = True,
        num_visual_tokens: int = 8,
        output_dim: int = 384,
        image_size: int = 224,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_visual_tokens = num_visual_tokens
        self.output_dim = output_dim
        self.image_size = image_size

        # Initialize backbone
        if model_name == "repvit_xxs":
            self.backbone = repvit_xxs(pretrained=pretrained)
            self.backbone_dim = 256  # XXS output dimension
        elif model_name.startswith("repvit_m"):
            # Use timm models directly
            import timm
            timm_model_name = f"{model_name}.dist_450e_in1k" if "dist" not in model_name else model_name
            self.backbone = timm.create_model(
                timm_model_name, pretrained=pretrained, num_classes=0)

            # Determine backbone output dimension based on model size
            # Note: These are the actual output dimensions from the timm models
            model_dims = {
                'repvit_m0_9': 384,  # Corrected: actual output is 384
                'repvit_m1_0': 384,  # Corrected: actual output is 384
                'repvit_m1_1': 384,
                'repvit_m1_5': 512,
                'repvit_m2_3': 640,
            }
            base_name = model_name.split('.')[0]
            self.backbone_dim = model_dims.get(base_name, 384)
            print(
                f"Using timm model: {timm_model_name} with output_dim={self.backbone_dim}")
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Use 'repvit_xxs' or timm model like 'repvit_m0_9'")

        # Adaptive pooling to get fixed number of tokens
        pool_size = int(num_visual_tokens ** 0.5)
        if pool_size ** 2 != num_visual_tokens:
            # Non-square pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))  # 8 tokens
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Layer norm for features
        self.ln_vision = nn.LayerNorm(self.backbone_dim)

        # Freeze backbone if specified
        if freeze:
            self._freeze_backbone()

        # Initialize projection
        self._init_weights()

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - visual_tokens: Visual embeddings [B, num_tokens, output_dim]
                - pooled_output: Global pooled features [B, output_dim]
        """
        batch_size = pixel_values.size(0)

        # Extract features from backbone
        with torch.set_grad_enabled(not self.backbone.training):
            features = self.backbone.forward_features(pixel_values)

        # features shape: [B, C, H, W]
        # Adaptive pooling first (before normalization)
        pooled_features = self.adaptive_pool(features)  # [B, C, h, w]

        # Reshape to sequence of tokens for normalization
        B, C, h, w = pooled_features.shape
        visual_tokens = pooled_features.permute(
            0, 2, 3, 1).reshape(B, h * w, C)

        # Apply layer norm in the correct dimension
        visual_tokens = self.ln_vision(visual_tokens)  # [B, h*w, C]

        # Project to output dimension
        visual_tokens = self.projection(visual_tokens)

        # Global pooled output (from original features, not pooled)
        pooled_output = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C]
        pooled_output = self.projection[0](pooled_output)

        if return_dict:
            return {
                'visual_tokens': visual_tokens,
                'pooled_output': pooled_output,
            }
        return visual_tokens

    def get_num_visual_tokens(self) -> int:
        """Return number of visual tokens."""
        return self.num_visual_tokens

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype


class MobileViTEncoder(nn.Module):
    """
    MobileViT Vision Encoder wrapper for EmberVLM.

    Uses official Apple MobileViT-XS (~2.3M params) from HuggingFace as an alternative to RepViT.
    Model: apple/mobilevit-x-small
    Handles image preprocessing, feature extraction, and adaptive pooling
    to produce a fixed number of visual tokens.

    MobileViT-XS output dimension is 384, matching RepViT and the default
    fusion module configuration.
    """

    def __init__(
        self,
        model_name: str = "apple/mobilevit-x-small",
        pretrained: bool = True,
        freeze: bool = True,
        num_visual_tokens: int = 8,
        output_dim: int = 384,
        image_size: int = 256,  # MobileViT default is 256
    ):
        super().__init__()

        try:
            from transformers import AutoModel, AutoImageProcessor
            HF_AVAILABLE = True
        except ImportError:
            HF_AVAILABLE = False
            AutoModel = None
            AutoImageProcessor = None

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for MobileViTEncoder. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.num_visual_tokens = num_visual_tokens
        self.output_dim = output_dim
        self.image_size = image_size

        # Load official Apple MobileViT from HuggingFace
        # Default: apple/mobilevit-x-small (~2.3M params)
        print(
            f"Loading official Apple MobileViT from HuggingFace: {model_name}...")

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=False,
            )
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        # MobileViT-XS output dimension - probe actual output to get correct dimension
        # Run a dummy forward pass to determine actual output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            dummy_output = self.backbone(dummy_input, output_hidden_states=True, return_dict=True)
            self.backbone_dim = dummy_output.last_hidden_state.shape[1]  # [B, C, H, W] -> get C
        print(f"Loaded MobileViT with backbone_dim={self.backbone_dim} (probed from output)")

        # Adaptive pooling to get fixed number of tokens
        pool_size = int(num_visual_tokens ** 0.5)
        if pool_size ** 2 != num_visual_tokens:
            # Non-square pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))  # 8 tokens
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Projection to output dimension
        # Always create projection layer (even if dims match) for consistent interface
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Layer norm for features
        self.ln_vision = nn.LayerNorm(self.backbone_dim)

        # Freeze backbone if specified
        if freeze:
            self._freeze_backbone()

        # Initialize projection
        self._init_weights()

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - visual_tokens: Visual embeddings [B, num_tokens, output_dim]
                - pooled_output: Global pooled features [B, output_dim]
        """
        import logging
        logger = logging.getLogger(__name__)

        batch_size = pixel_values.size(0)

        # DIAGNOSTIC: Log input pixel values variance
        if not hasattr(self, '_encoder_logged'):
            pv_mean = pixel_values.mean().item()
            pv_std = pixel_values.std().item()
            pv_var = pixel_values.var().item()
            logger.info(f"[VisionEncoder] INPUT pixel_values: shape={pixel_values.shape}, "
                       f"mean={pv_mean:.4f}, std={pv_std:.4f}, var={pv_var:.4f}")

        # Extract features from HuggingFace MobileViT backbone
        with torch.set_grad_enabled(not self.backbone.training):
            outputs = self.backbone(
                pixel_values, output_hidden_states=True, return_dict=True)
            # Get last hidden state from the convolutional layers
            # HuggingFace MobileViT returns last_hidden_state in [B, C, H, W] format
            features = outputs.last_hidden_state

        # DIAGNOSTIC: Log backbone output variance
        if not hasattr(self, '_encoder_logged'):
            feat_mean = features.mean().item()
            feat_std = features.std().item()
            feat_var = features.var().item()
            logger.info(f"[VisionEncoder] BACKBONE features: shape={features.shape}, "
                       f"mean={feat_mean:.4f}, std={feat_std:.4f}, var={feat_var:.4f}")

        # features shape: [B, C, H, W]
        # Adaptive pooling
        pooled_features = self.adaptive_pool(features)  # [B, C, h, w]

        # Reshape to sequence of tokens
        B, C, h, w = pooled_features.shape
        visual_tokens = pooled_features.permute(
            0, 2, 3, 1).reshape(B, h * w, C)

        # Apply layer norm
        visual_tokens = self.ln_vision(visual_tokens)  # [B, h*w, C]

        # Project to output dimension
        visual_tokens = self.projection(visual_tokens)

        # DIAGNOSTIC: Log final visual tokens variance
        if not hasattr(self, '_encoder_logged'):
            vt_mean = visual_tokens.mean().item()
            vt_std = visual_tokens.std().item()
            vt_var = visual_tokens.var().item()
            logger.info(f"[VisionEncoder] FINAL visual_tokens: shape={visual_tokens.shape}, "
                       f"mean={vt_mean:.4f}, std={vt_std:.4f}, var={vt_var:.4f}")
            self._encoder_logged = True

        # Global pooled output (from original features)
        pooled_output = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C]
        pooled_output = self.projection[0](pooled_output)

        if return_dict:
            return {
                'visual_tokens': visual_tokens,
                'pooled_output': pooled_output,
            }
        return visual_tokens

    def get_num_visual_tokens(self) -> int:
        """Return number of visual tokens."""
        return self.num_visual_tokens

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype


class DINOv2Encoder(nn.Module):
    """
    DINOv2-Small Vision Encoder for EmberVLM.

    Uses facebook/dinov2-small (22M params, 44MB FP16) for superior
    semantic features compared to MobileViT-XS.

    Key advantages:
    - Self-supervised pretraining on ImageNet-22K
    - 256 visual tokens (vs 64 for MobileViT) = 4x more spatial detail
    - Better object localization and semantic understanding
    - Proven for vision-language tasks (used in Idefics, LLaVA variants)

    Output: [B, 256, 384] visual tokens (16x16 grid from 224x224 image)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        pretrained: bool = True,
        freeze: bool = True,
        num_visual_tokens: int = 256,  # 16x16 patches for DINOv2
        output_dim: int = 384,
        image_size: int = 224,
    ):
        super().__init__()

        try:
            from transformers import Dinov2Model, Dinov2Config
            HF_AVAILABLE = True
        except ImportError:
            HF_AVAILABLE = False
            Dinov2Model = None

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for DINOv2Encoder. "
                "Install with: pip install transformers>=4.35.0"
            )

        self.model_name = model_name
        self.num_visual_tokens = num_visual_tokens
        self.output_dim = output_dim
        self.image_size = image_size

        # Load DINOv2-Small from HuggingFace
        logger.info(f"Loading DINOv2-Small from HuggingFace: {model_name}...")

        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(model_name)
        else:
            config = Dinov2Config.from_pretrained(model_name)
            self.backbone = Dinov2Model(config)

        # DINOv2-Small has hidden_size=384
        self.backbone_dim = self.backbone.config.hidden_size
        logger.info(f"Loaded DINOv2-Small with backbone_dim={self.backbone_dim}")

        # DINOv2 patch size is 14x14, so 224x224 image -> 16x16 = 256 patches.
        # forward() prepends CLS and patch_avg, producing 2 + num_patches tokens.
        expected_patches = (image_size // 14) ** 2
        # num_visual_tokens here refers to patch tokens only (before CLS/avg prepend).
        # The actual output length from forward() will be 2 + self.num_visual_tokens.
        if num_visual_tokens > expected_patches:
            logger.warning(
                f"Requested {num_visual_tokens} patch tokens but DINOv2 produces {expected_patches} patches. "
                f"Using {expected_patches} patch tokens (output will be {expected_patches + 2} with CLS+avg)."
            )
            self.num_visual_tokens = expected_patches

        # Adaptive pooling if we want fewer tokens than patches
        if num_visual_tokens < expected_patches:
            pool_size = int(num_visual_tokens ** 0.5)
            self.use_pooling = True
            self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            logger.info(f"Using adaptive pooling: {expected_patches} -> {num_visual_tokens} tokens")
        else:
            self.use_pooling = False
            self.adaptive_pool = None

        # Projection to output dimension (identity if dims match)
        if self.backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.projection = nn.LayerNorm(output_dim)

        # Freeze backbone if specified (RECOMMENDED for VLM training)
        if freeze:
            self._freeze_backbone()
            logger.info("✓ DINOv2 backbone frozen (only projection trainable)")

        # Initialize projection
        self._init_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"DINOv2Encoder: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DINOv2 vision encoder.

        Builds the visual sequence as: [CLS, patch_avg, patches] so that
        the language model receives both a global summary (CLS + patch_avg)
        and fine-grained spatial features (patch tokens).

        Args:
            pixel_values: Input images [B, C, H, W] normalized
            return_dict: Whether to return dictionary

        Returns:
            Dictionary containing:
                - visual_tokens: [B, 2+K, output_dim]  (CLS + patch_avg + K patch tokens)
                - pooled_output: [B, output_dim] global features (CLS + patch_avg pooled)
        """
        batch_size = pixel_values.size(0)

        # DINOv2 forward pass
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(
                pixel_values,
                return_dict=True,
            )

        # DINOv2 output: [B, 1 + num_patches, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0:1, :]         # [B, 1, hidden_size]
        patch_tokens = last_hidden_state[:, 1:, :]       # [B, N, hidden_size]
        patch_avg = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, hidden_size]

        # Optional: adaptive pooling applied only to patch_tokens
        if self.use_pooling and self.adaptive_pool is not None:
            num_patches = patch_tokens.size(1)
            dim = patch_tokens.size(2)
            h = w = int(num_patches ** 0.5)
            patch_tokens_grid = patch_tokens.reshape(batch_size, h, w, dim)
            patch_tokens_grid = patch_tokens_grid.permute(0, 3, 1, 2)       # [B, C, H, W]
            patch_tokens_pooled = self.adaptive_pool(patch_tokens_grid)     # [B, C, H', W']
            h_new, w_new = patch_tokens_pooled.shape[2], patch_tokens_pooled.shape[3]
            patch_tokens = patch_tokens_pooled.permute(0, 2, 3, 1).reshape(
                batch_size, h_new * w_new, dim
            )  # [B, K, hidden_size]

        # Build final visual sequence: [CLS, patch_avg, patch_tokens]
        vision_tokens = torch.cat([cls_token, patch_avg, patch_tokens], dim=1)  # [B, 2+K, hidden_size]
        visual_tokens = self.projection(vision_tokens)

        # Pooled output: element-wise average of CLS + patch_avg → [B, hidden_size]
        pooled_vec = (cls_token.squeeze(1) + patch_avg.squeeze(1)) / 2.0  # [B, hidden_size]
        pooled_output = self.projection(pooled_vec)

        if return_dict:
            return {
                'visual_tokens': visual_tokens,
                'pooled_output': pooled_output,
            }
        return visual_tokens

    def get_num_visual_tokens(self) -> int:
        """Return total tokens from forward(): patches + CLS + patch_avg."""
        return self.num_visual_tokens + 2

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype


def create_vision_encoder(
    backbone_type: str = VISION_BACKBONE_REPVIT,
    model_name: Optional[str] = None,
    pretrained: bool = True,
    freeze: bool = True,
    num_visual_tokens: int = 8,
    output_dim: int = 384,
    image_size: int = 224,
) -> nn.Module:
    """
    Factory function to create vision encoder.

    Args:
        backbone_type: Vision backbone type ('repvit', 'mobilevit_xs', or 'dinov2_small')
        model_name: Specific model name (overrides backbone_type defaults)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
        num_visual_tokens: Number of visual tokens to output
        output_dim: Output dimension for visual tokens
        image_size: Input image size

    Returns:
        Vision encoder module (RepViTEncoder, MobileViTEncoder, or DINOv2Encoder)
    """
    # DINOv2 - RECOMMENDED for best vision-language performance
    if backbone_type == VISION_BACKBONE_DINOV2_SMALL or (model_name and 'dinov2' in model_name.lower()):
        actual_model_name = model_name if model_name else "facebook/dinov2-small"
        # DINOv2 uses 14x14 patches, so 224x224 -> 256 patch tokens.
        # forward() prepends CLS + patch_avg = 258 total, but the encoder
        # __init__ expects the *patch* count only (256); it handles the +2
        # inside forward().
        expected_patches = (image_size // 14) ** 2
        # If caller sends the full 258 (config.num_visual_tokens), strip 2
        if num_visual_tokens == expected_patches + 2:
            patch_tokens = expected_patches
        elif num_visual_tokens <= 8:
            # Legacy default — use all patches
            patch_tokens = expected_patches
        else:
            patch_tokens = num_visual_tokens
        logger.info(f"Creating DINOv2Encoder with {patch_tokens} patch tokens (output will be {patch_tokens + 2} with CLS+avg)")
        return DINOv2Encoder(
            model_name=actual_model_name,
            pretrained=pretrained,
            freeze=freeze,
            num_visual_tokens=patch_tokens,
            output_dim=output_dim,
            image_size=image_size,
        )
    # MobileViT-XS
    elif backbone_type == VISION_BACKBONE_MOBILEVIT_XS or (model_name and 'mobilevit' in model_name.lower()):
        actual_model_name = model_name if model_name else "apple/mobilevit-x-small"
        # MobileViT default image size is 256
        actual_image_size = image_size if image_size != 224 else 256
        return MobileViTEncoder(
            model_name=actual_model_name,
            pretrained=pretrained,
            freeze=freeze,
            num_visual_tokens=num_visual_tokens,
            output_dim=output_dim,
            image_size=actual_image_size,
        )
    # RepViT (default)
    else:
        actual_model_name = model_name if model_name else "repvit_m0_9"
        return RepViTEncoder(
            model_name=actual_model_name,
            pretrained=pretrained,
            freeze=freeze,
            num_visual_tokens=num_visual_tokens,
            output_dim=output_dim,
            image_size=image_size,
        )


# Image preprocessing utilities
class ImagePreprocessor:
    """Preprocessor for vision encoder input."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, images) -> torch.Tensor:
        """
        Preprocess images for vision encoder.

        Args:
            images: Raw images - can be:
                - torch.Tensor [B, C, H, W] or [C, H, W] in range [0, 1]
                - PIL.Image.Image (will be converted to tensor)

        Returns:
            Normalized images ready for encoder [C, H, W] or [B, C, H, W]
        """
        from PIL import Image
        import numpy as np

        # Handle PIL Image input
        if isinstance(images, Image.Image):
            # Convert PIL Image to tensor [C, H, W]
            img_array = np.array(images).astype(np.float32) / 255.0
            if img_array.ndim == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=0)
            elif img_array.ndim == 3:  # RGB
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
            images = torch.from_numpy(img_array)

        # Ensure tensor
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or PIL.Image, got {type(images)}")

        # Add batch dimension if needed
        original_dims = images.dim()
        if images.dim() == 3:
            images = images.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        # Resize if needed
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )

        # Normalize
        mean = torch.tensor(
            self.mean, device=images.device, dtype=images.dtype)
        std = torch.tensor(self.std, device=images.device, dtype=images.dtype)
        images = (images - mean[None, :, None, None]) / \
            std[None, :, None, None]

        # Remove batch dimension if input was 3D
        if original_dims == 3:
            images = images.squeeze(0)  # [1, C, H, W] -> [C, H, W]

        return images
