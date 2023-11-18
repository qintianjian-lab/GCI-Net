import torch
import torch.nn as nn
import math
import  time

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
            x = x / keep_prob * mask
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x



class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.conv0_3 = nn.Conv3d(dim, dim, (1, 1, 7), padding=(0, 0, 3), groups=dim)


        self.conv1_1 = nn.Conv3d(dim, dim, (1, 11, 1), padding=(0, 5, 0), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim)
        self.conv1_3 = nn.Conv3d(dim, dim, (1, 1, 11), padding=(0, 0, 5), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (1, 21, 1), padding=(0, 10, 0), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (21, 1, 1), padding=(10, 0, 0), groups=dim)
        self.conv2_3 = nn.Conv3d(
            dim, dim, (1, 1, 21), padding=(0, 0, 10), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.conv1_3(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.conv2_3(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)#TODO: 1*1

        return attn * u

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        # self.norgciunet = build_norm_layer(norm_cfg, dim)[1]
        self.norgciunet = nn.BatchNorm3d(dim,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.attn = SpatialAttention(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = nn.BatchNorm3d(dim,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        # start_time = time.time()
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norgciunet(x)))
        # end_time = time.time() - start_time
        # print('attn_time:', end_time)

        # start_time = time.time()
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        # end_time = time.time() - start_time
        # print('mlp_time:', end_time)
        # x = x.view(B, C, N).permute(0, 2, 1)
        return x


# @ BACKBONES.register_module()
class MSCAC(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=int,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depth=4,

                 # norm_cfg=dict(type='BN3d', requires_grad=True),
                 ):
        super(MSCAC, self).__init__()


        self.depth = depth


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule

        blocks = [Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[j])
                  for j in range(self.depth)]

        self.block = nn.Sequential(*blocks)
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = nn.BatchNorm3d(embed_dim,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                std = math.sqrt(2.0 / fan_out)
                nn.init.normal_(m.weight, mean=0., std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        B, C, H, W, D = x.shape

        # x = x.reshape(B, C, H * W * D).permute(0, 2, 1)  #(B, C, H, W, D) => (B, N, C)


        for blk in self.block:
            x = blk(x)


        # x = x.reshape(B, C, H * W * D).permute(0, 2, 1)  #(B, C, H, W, D) => (B, N, C)
        #
        x = self.norm(x)
        # x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)

        return x
