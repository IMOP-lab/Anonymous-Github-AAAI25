import torch.nn as nn
from monai.networks.nets import BasicUNet
from monai.networks.nets import UNETR
from networks.PaR.PaR import PaR

class UNet3D_PaR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Initialize the 3D-UNet
        self.UNet3D = BasicUNet(in_channels=in_channels, out_channels=out_channels)

        self.par = PaR(axial_dim=96, in_channels=self.out_channels, heads=8, groups=8)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x1 = self.UNet3D(x)
        x2 = self.par(x1)
        x3 = self.sigmoid(x2)
        out = x3 * x1 + x1
        return out
    
class UNETR_PaR(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, res_block, dropout_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Initialize the UNETR
        self.UNETR = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate)

        self.par = PaR(axial_dim=96, in_channels=self.out_channels, heads=8, groups=8)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x1 = self.UNETR(x)
        x2 = self.par(x1)
        x3 = self.sigmoid(x2)
        out = x3 * x1 + x1
        return out