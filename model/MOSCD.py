import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders.resnet import (ResNetEncoder, resnet_encoders)
from segmentation_models_pytorch.encoders.efficientnet import (EfficientNetEncoder, efficient_net_encoders)
from einops import repeat, rearrange


class ChangeDetectionHead(nn.Module):
    def __init__(
        self,
        in_channels = 128,
        inner_channels = 16,
        num_convs = 4,
        upsampling = 4,
        dilation = 1,
        fusion = 'diff',
    ):
        super(ChangeDetectionHead, self).__init__()
        if fusion == 'diff':
            in_channels = in_channels
            inner_channels = in_channels
        elif fusion == 'concat':
            in_channels = in_channels * 2
            inner_channels = in_channels
        layers = []
        if num_convs > 0:
            layers = [
                nn.modules.Sequential(
                    nn.modules.Conv2d(in_channels, inner_channels, 3, 1, 1, dilation=dilation),
                    nn.modules.BatchNorm2d(inner_channels),
                    nn.modules.ReLU(True),
                )
            ]
            if num_convs >  1:
                layers += [
                    nn.modules.Sequential(
                        nn.modules.Conv2d(inner_channels, inner_channels, 3, 1, 1, dilation=dilation),
                        nn.modules.BatchNorm2d(inner_channels),
                        nn.modules.ReLU(True),
                    )
                    for _ in range(num_convs - 1)
                ]

        cls_layer = nn.modules.Conv2d(inner_channels, 1, 3, 1, 1)
        layers.append(cls_layer) 
        self.convs = nn.modules.Sequential(*layers)
        
        self.upsampling_scale_factor = upsampling
        self.upsampling = nn.modules.UpsamplingBilinear2d(scale_factor=upsampling) if self.upsampling_scale_factor > 1 else nn.Identity()
        
    def forward(self, x):  
        x = self.convs(x)
        x_upsampling = self.upsampling(x)
        return torch.tensor(0), x_upsampling


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=128, proj_dim=64):
        super(ProjectionHead, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, in_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.proj(x) 



class Squeeze2(nn.Module):
    def forward(self, x):
        return x.squeeze(dim=2)

class TMEBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()        
        self.block = nn.Sequential(
            nn.modules.Conv2d(in_channel, out_channel, 3, 1, 1, dilation=1),
            nn.modules.BatchNorm2d(out_channel),
            nn.modules.ReLU(True)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x


class TME(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TME, self).__init__()   
        
        self.tempatt_list = nn.ModuleList([TMEBlock(in_channel, out_channel) for in_channel, out_channel in zip(in_channels, out_channels)])
        
    def forward(self, features_A, features_B):
        TempAtt_features_A = [tempatt(fa) for tempatt, fa in zip(self.tempatt_list, features_A)]
        TempAtt_features_B = [tempatt(fb) for tempatt, fb in zip(self.tempatt_list, features_B)]
        tempatt_features = [fab * fba for fab, fba in zip(TempAtt_features_A, TempAtt_features_B)]
        return tempatt_features
    


class MOSCD(nn.Module):
    def __init__(self, args):
        super(MOSCD, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        # encoder
        if 'res' in self.args.backbone:
            encoder_params = resnet_encoders[args.backbone]['params']
            self.encoder = ResNetEncoder(**encoder_params)
        elif 'eff' in self.args.backbone:
            encoder_params = efficient_net_encoders[args.backbone]['params']
            self.encoder = EfficientNetEncoder(**encoder_params)
        
        # seg decoder
        self.seg_decoder = FPNDecoder(
            encoder_channels = encoder_params['out_channels'],
            pyramid_channels = 256,
            segmentation_channels = 128,
        )
        # seg head
        upsampling = 1 if args.downsample_seg else 4
        self.head_seg = SegmentationHead(
            in_channels = 128,
            out_channels = self.args.num_segclass,
            upsampling = upsampling   
        ) 
        
        # feat-bcd branch
        self.tempAtt = TME(
            in_channels = encoder_params['out_channels'],
            out_channels = encoder_params['out_channels'],
        )
        self.bcd_decoder = FPNDecoder(
            encoder_channels = encoder_params['out_channels'],
            pyramid_channels = 256,
            segmentation_channels = 128,
        )
        self.head_bcd_feat = SegmentationHead(
            in_channels = 128,
            out_channels = self.args.num_segclass,
            upsampling = upsampling # 1 
        ) 

        # merge-bcd
        upsampling_bcd = 4 if args.downsample_seg else 1
        self.head_bcd = ChangeDetectionHead(
            in_channels = self.args.num_segclass,
            inner_channels = self.args.num_segclass,
            num_convs = self.args.bcd_convs_num,
            upsampling = upsampling_bcd, 
            fusion = self.args.fusion
        ) 

        
        self.proj_head = ProjectionHead(
            in_channels = 128,
            proj_dim = 32
        )
        
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4)
        
        if self.args.pretrained:            
            self._init_weighets()
        else:
            self._init_weighets_kaiming(self.encoder, self.decoder, self.head_bcd, self.head_seg, self.proj_head)
        
    def _init_weighets(self):
        '''
        pretrained weights can be refered to torchvision.
        '''
        if self.args.backbone == 'resnet18':
            encoder_pred = torchvision.models.resnet18()
            pre = torch.load('./pretrain/resnet18-f37072fd.pth')
            encoder_pred.load_state_dict(pre)
        elif self.args.backbone == 'resnet34':
            encoder_pred = torchvision.models.resnet34()
            pre = torch.load('./pretrain/resnet34-b627a593.pth')
            encoder_pred.load_state_dict(pre)
        elif self.args.backbone == 'resnet50':    
            encoder_pred = torchvision.models.resnet50(weights='./pretrain/resnet50-0676ba61.pth')  
        elif self.args.backbone == 'efficientnet-b0':
            encoder_pred = torchvision.models.efficientnet_b0()
            pre = torch.load('./pretrain/efficientnet_b0_rwightman-3dd342df.pth')
            encoder_pred.load_state_dict(pre)
        elif self.args.backbone == 'efficientnet-b1':
            encoder_pred = torchvision.models.efficientnet_b1()  
            pre = torch.load('./pretrain/efficientnet_b1_rwightman-533bc792.pth')
            encoder_pred.load_state_dict(pre)
        elif self.args.backbone == 'efficientnet-b2':
            encoder_pred = torchvision.models.efficientnet_b2()  
            pre = torch.load('./pretrain/efficientnet_b2_rwightman-bcdf34b7.pth')
            encoder_pred.load_state_dict(pre)
        elif self.args.backbone == 'efficientnet-b3':
            encoder_pred = torchvision.models.efficientnet_b3()  
            pre = torch.load('./pretrain/efficientnet_b3_rwightman-cf984f9c.pth')
            encoder_pred.load_state_dict(pre)
                       
        pretrained_dict = encoder_pred.state_dict()
        encoder_dict = self.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)   
        
    def _init_weighets_kaiming(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_() 
                    
    

    def forward(self, img_A, img_B=None, hook=None):
        if hook == 'seg':
            features_A = self.encoder(img_A)
            seg_A = self.seg_decoder(*features_A)
            logits_A = self.head_seg(seg_A)
            outputs = {}
            if self.args.downsample_seg:
                logits_A = self.upsampling(logits_A)
            outputs['seg_A'] = logits_A
            return outputs
        else:
            features_A = self.encoder(img_A)
            features_B = self.encoder(img_B)
            # seg
            seg_A = self.seg_decoder(*features_A)
            seg_B = self.seg_decoder(*features_B)
            logits_A = self.head_seg(seg_A)
            logits_B = self.head_seg(seg_B) 
                
            # seg-bcd
            if self.args.fusion == 'diff':
                logits_AB = torch.abs(logits_A - logits_B)
            elif self.args.fusion == 'concat':
                logits_AB = torch.concat([logits_A, logits_B], dim=1)
            
            # feat bcd
            temaatt_features = self.tempAtt(features_A, features_B)
            logits_featBCD = self.bcd_decoder(*temaatt_features) 
            logits_featBCD = self.head_bcd_feat(logits_featBCD)

            # CBC
            conc_logits_BCD = logits_AB * logits_featBCD
            _, logits_BCD = self.head_bcd(conc_logits_BCD)

            outputs = {}            
            if self.args.downsample_seg:
                logits_A = self.upsampling(logits_A)
                logits_B = self.upsampling(logits_B)
            
            outputs['seg_A'] = logits_A
            outputs['seg_B'] = logits_B
            outputs['BCD'] = logits_BCD     
            return outputs
   
        
if __name__ == '__main__':
    pass       