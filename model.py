"""
Network initialization library where you can add or modify any 3D segmentation network.

Create on 2024-6-1 Saturday.   

"""

from monai.networks.nets import BasicUNet
from monai.networks.nets.unet import UNet as Monai_UNet
from monai.networks.nets import VNet as Monai_VNet
from monai.networks.nets import SegResNet
from monai.networks.nets import UNETR
from highresnet import HighRes3DNet
from networks.UXNet_3D.network_backbone import UXNET
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.SwinUNETR.SwinUNETR import SwinUNETR
from networks.PaR.backbone_PaR import UNet3D_PaR, UNETR_PaR

def get3dmodel(network, in_channel, out_classes):
    ## UNet
    if network == 'UNet':
        model = BasicUNet(in_channels=in_channel, out_channels=out_classes)
        
    ## Monai_UNet
    elif network == 'Monai_Unet':
        model = Monai_UNet(
            spatial_dims=3, 
            in_channels=in_channel, 
            out_channels=out_classes, 
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2))
        
    ## VNet
    elif network == 'Vnet':
        model = Monai_VNet(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_classes)
    
    ## SegResNet
    elif network == 'SegResNet':
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_classes,
            init_filters=16,
            dropout_prob=0.5)
        
    ## UNETR
    elif network == 'UNETR':
        model = UNETR(
            in_channels=in_channel,
            out_channels=out_classes,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0)
        
    ## HighRes3DNet
    elif network == 'HighRes3DNet':
        model = HighRes3DNet(
            in_channels=in_channel, 
            out_channels=out_classes)
        
    ## 3DUXNET
    elif network == '3DUXNET':
        model = UXNET(
            in_chans=in_channel,
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3)
  
    ## nnFormer
    elif network == 'nnFormer':
        model = nnFormer(
            input_channels=in_channel, 
            num_classes=out_classes)      
        
    ## TransBTS
    elif network == 'TransBTS':
        _, model = TransBTS(img_dim=96,num_classes = out_classes , _conv_repr=True, _pe_type='learned')
        
    ## SwinUNETR 
    elif network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False)   
        
    ## UNet+PaR
    elif network=="UNet3D_PaR": 
        model = UNet3D_PaR(
            in_channels = in_channel,
            out_channels = out_classes,)    
        
    ## UNet+PaR
    elif network=="UNETR_PaR": 
        model = UNETR_PaR(
            in_channels=in_channel,
            out_channels=out_classes,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0)
    return model
