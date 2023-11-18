from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from gciunet.network_architecture.layers import LayerNorm
from gciunet.network_architecture.dynunet_block import get_conv_layer, UnetResBlock,SeparableConv3d,EmbeddingBraTS
from gciunet.network_architecture.MSCAM import MSCAC

import torch

import math


einops, _ = optional_import("einops")

class GCIUNetEncoder(nn.Module):

    def __init__(self, dims=[32, 64, 128, 256], msca_dim=[32, 64, 128], spatial_dims=3, in_channels=4, dropout=0.0):
        """
            Args:
                dims: number of channel maps for the stages.
                msca_dim: number of channels for the stages in MSCAC.
                spatial_dims: dimension of the input image.
                in_channels: dimension of input channels.
                dropout: faction of the input units to drop.
        """

        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            EmbeddingBraTS(in_channels=in_channels,out_channels=dims[0]),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)



        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        # Only for the first three accessions to 3D-MSCAC
        for i in range(3):
            stage_blocks = []
            stage_blocks.append(MSCAC(embed_dim=msca_dim[i]))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        hidden_states = []
        for i in range(0, 3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            hidden_states.append(x)
        x = self.downsample_layers[3](x)
        x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        hidden_states.append(x)
        return x, hidden_states


    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


# convCA used in FusionBlock
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels * 2, in_channels * 2 // reduction_ratio)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_channels * 2 // reduction_ratio, in_channels * 2)
        self.sigmoid = nn.LeakyReLU()

    def forward(self, x):
        batch_size, num_channels, _, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1, 1)
        return x * y.expand_as(x)

# used in Last Deocder stage
class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FusionBlock, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.conv1 = get_conv_layer(
            spatial_dims=3,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=1, stride=1, dropout=0.0, bias=True, conv_only=True)

    def forward(self, x1, x2):
        x_all = x1 + x2
        x = torch.cat([x1, x2], dim=1)
        x = self.ca(x)
        x = self.conv1(x)
        out = x_all + x
        return out


# Replaces the ConvBlock(UnetResBlock) used in the last decoder
class FinalConv(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = SeparableConv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.conv2 = SeparableConv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.relu(x) + residual
        return out

# Last UpBlock in Deocder of GCI-UNet
class LastGCIUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            upsample_kernel_size: Union[Sequence[int], int],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.finalconv = FinalConv(out_channels)
        self.fusion = FusionBlock(out_channels)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        res = self.fusion(out, skip)
        res = self.finalconv(res)
        return res

# Global-Guided Feature Fusion in GFES
class GlobalGuidedFusion(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            gf_proj_size: int,
            skip_proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            gf_proj_size: projection size for global features in the GlobalGuidedLayer,
            skip_proj_size: projection size for features from encoder in the GlobalGuidedLayer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.norm_g = nn.LayerNorm(256)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.global_layer = GlobalGuidedLayer(input_size=input_size, gf_proj_size=gf_proj_size,
                                              skip_proj_size=skip_proj_size, num_heads=num_heads)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))


        self.pos_embed = None
        # add position embedding
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
            self.global_pos_embed = nn.Parameter(torch.zeros(1, 64, 256))


    def forward(self, skip, out, global_fea):

        B, C, H, W, D = skip.shape
        skip = skip.reshape(B, C, H * W * D).permute(0, 2, 1)
        out = out.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            skip = skip + self.pos_embed
            out = out + self.pos_embed
            global_fea = global_fea + self.global_pos_embed
        attn = self.gamma * self.global_layer(self.norm(skip), self.norm(out), self.norm_g(global_fea))
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)
        return x


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# Used in GlobalGuidedFusion
class GlobalGuidedLayer(nn.Module):

    def __init__(self, input_size, gf_proj_size, skip_proj_size, num_heads, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.EF1 = nn.Parameter(init_(torch.zeros(input_size, skip_proj_size)))  # 针对 global_fea，N 为4*4*4
        self.EF2 = nn.Parameter(init_(torch.zeros(64, gf_proj_size)))  # 针对 global_fea，N 为4*4*4
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, skip, out, global_fea):

        B, N, C = skip.shape
        B1, N1, C1 = global_fea.shape  # (1,64,256)
        out_v = out.reshape(B, N, self.num_heads, C // self.num_heads) #(B,512,4,32)
        skip_k = skip.reshape(B, N, self.num_heads, C // self.num_heads) #(B,512,4,32)
        global_q = global_fea.reshape(B1, N1, self.num_heads, C1 // self.num_heads) #(B,64,4,64)
        out_v = out_v.permute(0, 2, 3, 1) # (1,128,8,8,8) -> (1,4,32, 512)
        skip_k = skip_k.permute(0, 2, 3, 1)  # (1,128,8,8,8) -> (1,4,32, 512)
        global_q = global_q.permute(0, 2, 3, 1)  # （1,4,64,64） B  num C//num  N
        skip_k_projected = torch.einsum('bhdn,nk->bhdk', skip_k, self.EF1)  # (1,4,32,64)
        global_q_projected = torch.einsum('bhdn,nk->bhdk', global_q, self.EF2)  # (1,4,64,32)
        global_q_projected = torch.nn.functional.normalize(global_q_projected, dim=-1)
        skip_k_projected = torch.nn.functional.normalize(skip_k_projected, dim=-1)

        attn = (skip_k_projected @ global_q_projected) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ out_v).permute(0, 3, 1, 2).reshape(B, N, C) # before permute is (1, 4, 32, 512) ->(1,512,4,32)->(1,512,128)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

# Up-sampling block in the Decoder
class GCIUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            gf_proj_size: int,
            num_heads: int = 4,
            out_size: int = 0,
            skip_proj_size: int = 64,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            gf_proj_size: projection size for global features in the GlobalGuidedLayer,
            num_heads: number of heads inside each global-guided feature fusion module.
            out_size: spatial size for each decoder.
            skip_proj_size: projection size for features from encoder in the GlobalGuidedLayer.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )


        self.global_block = GlobalGuidedFusion(input_size=out_size, hidden_size=out_channels,
                                 gf_proj_size=gf_proj_size, skip_proj_size=skip_proj_size,
                                 num_heads=num_heads,
                                 dropout_rate=0.1, pos_embed=True)
        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)

        self.decoder_block = nn.ModuleList()

        self.decoder_block.append(MSCAC(embed_dim=out_channels))

        self.skipconv = UnetResBlock(
            spatial_dims=3,
            in_channels=out_channels,
            out_channels=out_channels*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.channelconv = get_conv_layer(
            spatial_dims, out_channels*2, out_channels, kernel_size=3, stride=1, dropout=0.1, bias=True,
            conv_only=True)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip, global_fea):

        x_up = self.transp_conv(inp)
        skip_double_channel = self.skipconv(skip)
        skip_out = self.channelconv(skip_double_channel)
        attn_sa = self.global_block(skip_out, x_up, global_fea)+x_up+skip_out
        out = self.decoder_block[0](attn_sa)

        return out