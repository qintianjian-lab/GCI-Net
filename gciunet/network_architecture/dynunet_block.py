from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
            trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int,
            dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = False,
        conv_only: bool = True,
        is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
        kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
        kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=0, dilation=(1, 1, 1),
                 bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, (1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# Patch Embedding Layer in BraTS
class EmbeddingBraTS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.conv1 = SeparableConv3d(in_channels, in_channels * 4, (3, 3, 3), (2, 2, 2), padding=get_padding(3, 2),
                                     bias=False)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(in_channels * 4)
        self.conv2 = SeparableConv3d(in_channels * 4, in_channels * 4, (3, 3, 3), (1, 1, 1), padding=get_padding(3, 1),
                                     bias=False)
        self.norm2 = nn.LayerNorm(in_channels * 4)
        self.conv3 = SeparableConv3d(in_channels * 4, out_channels, (3, 3, 3), (2, 2, 2), padding=get_padding(3, 2),
                                     bias=False)
        self.norm3 = nn.LayerNorm(out_channels)
        self.conv4 = SeparableConv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), padding=get_padding(3, 1),
                                     bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)  # (1, 8, 64, 64, 64)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)  # 64, 64, 64
        x = x.flatten(2).transpose(1, 2).contiguous()  # x.flatten(2) -> (1, 8, 262144)  x.transpose(1, 2) -> (1, 262144, 8)
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.in_dim * 4, Ws, Wh, Ww)
        x = self.conv2(x)
        x = self.gelu(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.in_dim * 4, Ws, Wh, Ww)
        x = self.conv3(x)
        x = self.gelu(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm3(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        x = self.conv4(x)

        return x

# Patch Embedding Layer in ACDC
class EmbeddingACDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.conv1 = SeparableConv3d(in_channels, in_channels * 4, (1, 3, 3), (1, 2, 2),
                                     padding=get_padding([1, 3, 3], [1, 2, 2]), bias=False)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(in_channels * 4)
        self.conv2 = SeparableConv3d(in_channels * 4, in_channels * 4, (1, 3, 3), (1, 1, 1),
                                     padding=get_padding([1, 3, 3], [1, 1, 1]), bias=False)
        self.norm2 = nn.LayerNorm(in_channels * 4)
        self.conv3 = SeparableConv3d(in_channels * 4, out_channels, (1, 3, 3), (1, 2, 2),
                                     padding=get_padding([1, 3, 3], [1, 2, 2]), bias=False)
        self.norm3 = nn.LayerNorm(out_channels)
        self.conv4 = SeparableConv3d(out_channels, out_channels, (1, 3, 3), (1, 1, 1),
                                     padding=get_padding([1, 3, 3], [1, 1, 1]), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)  # (1, 8, 64, 64, 64)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)  # 64, 64, 64
        x = x.flatten(2).transpose(1,2).contiguous()  # x.flatten(2) -> (1, 8, 262144)  x.transpose(1, 2) -> (1, 262144, 8)
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.in_dim * 4, Ws, Wh, Ww)
        x = self.conv2(x)
        x = self.gelu(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.in_dim * 4, Ws, Wh, Ww)

        x = self.conv3(x)
        x = self.gelu(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm3(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv4(x)

        return x


# Global-Local contrast loss
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, enc_feature, dec_feature):
        enc_norm = torch.nn.functional.normalize(enc_feature, dim=1)
        dec_norm = torch.nn.functional.normalize(dec_feature, dim=1)
        similarity = torch.sum(enc_norm * dec_norm, dim=1)
        mean_similarity = torch.mean(similarity)
        return 1 - mean_similarity


# Change in feature dimension of global features before contrast loss
class UpsampleBlock(nn.Module):
    def __init__(self, dims, in_channels):
        super(UpsampleBlock, self).__init__()

        self.channelconv1 = nn.Conv3d(in_channels, dims[0], (1, 1, 1))
        self.channelconv2 = nn.Conv3d(in_channels, dims[1], (1, 1, 1))

        self.transp_conv_1 = get_conv_layer(
            3,
            in_channels=dims[0],
            out_channels=dims[0],
            kernel_size=4,
            stride=4,
            conv_only=True,
            is_transposed=True,
        )

        self.transp_conv_2_1 = get_conv_layer(
            3,
            in_channels=dims[1],
            out_channels=dims[1],
            kernel_size=4,
            stride=4,
            conv_only=True,
            is_transposed=True,
        )

        self.transp_conv_2_2 = get_conv_layer(
            3,
            in_channels=dims[1],
            out_channels=dims[1],
            kernel_size=2,
            stride=2,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, global_F):
        channel64 = self.channelconv1(global_F)  # (1,64,4,4,4)
        channel32 = self.channelconv2(global_F)  # (1,32,4,4,4）

        # Use transposed convolution to upsample global
        f_64_16 = self.transp_conv_1(channel64)
        f_32_16 = self.transp_conv_2_1(channel32)
        f_32_32 = self.transp_conv_2_2(f_32_16)

        f_64_16 = torch.nn.functional.normalize(f_64_16, p=2, dim=1)
        f_32_32 = torch.nn.functional.normalize(f_32_32, p=2, dim=1)

        return f_64_16, f_32_32


