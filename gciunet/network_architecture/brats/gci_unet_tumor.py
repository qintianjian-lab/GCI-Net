from torch import nn
from typing import Tuple, Union
from gciunet.network_architecture.neural_network import SegmentationNetwork
from gciunet.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock,UpsampleBlock,CosineSimilarityLoss
from gciunet.network_architecture.brats.model_components import GCIUNetEncoder, GCIUpBlock, LastGCIUpBlock
import torch


class GCI_UNet(SegmentationNetwork):
    """
    GCI-UNET based on: "Qiao et al.,
    Less is More: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (4, 4, 4,)
        self.hidden_size = hidden_size

        self.gciunet_encoder = GCIUNetEncoder(dims=dims)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = GCIUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
            gf_proj_size=32,
        )
        self.decoder4 = GCIUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
            gf_proj_size=16,

        )
        self.decoder3 = GCIUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
            gf_proj_size=8,
        )
        self.decoder2 = LastGCIUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=(4, 4, 4),
        )

        self.skiploss = CosineSimilarityLoss()
        self.upsample_block =  UpsampleBlock(dims=[64,32],in_channels=256)


        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):

        x_output, hidden_states = self.gciunet_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)  # (1, 256, 4, 4, 4)

        '''
        Add x_output to be used as a bootstrap for global features,
        with different dimensional transformations for different layers of skip
        '''

        dec3 = self.decoder5(dec4, enc3, x_output)  # (1, 128, 8, 8, 8)
        dec2 = self.decoder4(dec3, enc2, x_output)  # (1, 64, 16, 16,16)
        dec1 = self.decoder3(dec2, enc1, x_output)  # (1, 32, 32, 32,32)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        # Setting loss weights for different layers
        # Processing the global feature upsampling with dec3,dec2
        transpose_2, transpose_3 = self.upsample_block(dec4)  # The shape of transpose_2 is (1, 64, 16, 16,16) and transpose_3 is (1, 32, 32, 32,32)

        weight_loss2 = 0.3
        weight_loss3 = 0.7

        # Calculate each loss
        dec2 = torch.nn.functional.normalize(dec2, p=2, dim=1)
        dec1 = torch.nn.functional.normalize(dec1, p=2, dim=1)

        loss2 = weight_loss2 * self.skiploss(transpose_2, dec2)
        loss3 = weight_loss3 * self.skiploss(transpose_3, dec1)

        total_loss = loss2 + loss3
        return logits, total_loss
