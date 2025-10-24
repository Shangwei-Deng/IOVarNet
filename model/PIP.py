import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################

from .fusion import *

from .pde import pde
from thop import profile
##################################################

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.GELU()
        self.relu_s1 = nn.ReLU(inplace=True)
        # self.relu_s1 = nn.SiLU(inplace=True)


    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):
    _, _, hei, wid = tar.shape
    src = F.interpolate(src, size=[hei, wid], mode='bilinear', align_corners=True)


    return src


### RSU-3 ###
class PDEU3(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(PDEU3,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,mid_ch,dirate=1)
        self.in_pde_conv= REBNCONV(in_ch, out_ch=1, dirate=1)
        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv1 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv3d = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.fuse3 = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.fuse2 = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.fuse1 = REBNCONV(mid_ch*2,out_ch,dirate=1)


        self.fcl1 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl2 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl3 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl3f = nn.Conv2d(mid_ch, 1, 1)
        self.fcl2f = nn.Conv2d(mid_ch, 1, 1)
        self.fcl1f = nn.Conv2d(out_ch, 1, 1)

        self.tfd1 = pde()
        self.tfd2 = pde()
        self.tfd3 = pde()
        self.tfd3f = pde()
        self.tfd2f = pde()
        self.tfd1f = pde()

        self.out_conv = REBNCONV(out_ch, out_ch, 1)

    def forward(self,x):
        _, _, hei, wid = x.shape
        hx = x

        x_p = self.in_pde_conv(hx)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)


        # ------------decoder-----------

        hx3d = self.rebnconv3d(hx)
        hx3dup = _upsample_like(hx3d, hx3)
        hx3f = self.fuse3(torch.cat((hx3dup, hx3),1))

        hx2d = self.rebnconv2d(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2),1))

        hx1d = self.rebnconv1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1),1))


        gf1 = self.fcl1(hx1)
        gf2 = F.interpolate(self.fcl2(hx2), size=[hei, wid], mode='bilinear', align_corners=True)
        gf3 = F.interpolate(self.fcl3(hx3), size=[hei, wid], mode='bilinear', align_corners=True)
        gf3f = F.interpolate(self.fcl3f(hx3f), size=[hei, wid], mode='bilinear', align_corners=True)
        gf2f = F.interpolate(self.fcl2f(hx2f), size=[hei, wid], mode='bilinear', align_corners=True)
        gf1f = self.fcl1f(hx1f)

        s1 = self.tfd1(x_p, gf1)
        s2 = self.tfd2(s1, gf2)
        s3 = self.tfd3(s2, gf3)
        s3f = self.tfd3f(s3, gf3f)
        s2f = self.tfd2f(s3f, gf2f)
        s1f = self.tfd1f(s2f, gf1f)

        pde_pred = s1f

        # Out

        hx_out = hx1f + hx1f * pde_pred

        out = self.out_conv(hx_out)

        return out
    
### RSU-2 ###
class PDEU2(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(PDEU2,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,mid_ch,dirate=1)
        self.in_pde_conv= REBNCONV(in_ch, out_ch=1, dirate=1)
        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv1 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
 

        self.rebnconv2d = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch,mid_ch,dirate=1)


        self.fuse2 = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.fuse1 = REBNCONV(mid_ch*2,out_ch,dirate=1)


        self.fcl1 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl2 = nn.Conv2d(mid_ch, 1, 1)

        self.fcl2f = nn.Conv2d(mid_ch, 1, 1)
        self.fcl1f = nn.Conv2d(out_ch, 1, 1)

        self.tfd1 = pde()
        self.tfd2 = pde()

        self.tfd2f = pde()
        self.tfd1f = pde()

        self.out_conv = REBNCONV(out_ch, out_ch, 1)

    def forward(self,x):
        _, _, hei, wid = x.shape
        hx = x

        x_p = self.in_pde_conv(hx)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        # ------------decoder-----------

 

        hx2d = self.rebnconv2d(hx)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2),1))

        hx1d = self.rebnconv1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1),1))


        gf1 = self.fcl1(hx1)
        gf2 = F.interpolate(self.fcl2(hx2), size=[hei, wid], mode='bilinear', align_corners=True)
        gf2f = F.interpolate(self.fcl2f(hx2f), size=[hei, wid], mode='bilinear', align_corners=True)
        gf1f = self.fcl1f(hx1f)

        s1 = self.tfd1(x_p, gf1)
        s2 = self.tfd2(s1, gf2)
        s2f = self.tfd2f(s2, gf2f)
        s1f = self.tfd1f(s2f, gf1f)

        pde_pred = s1f

        # Out

        hx_out = hx1f + hx1f * pde_pred

        out = self.out_conv(hx_out)

        return out,s1,s2,s2f,s1f


class PDEU3d(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(PDEU3d,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,mid_ch,dirate=1)
        self.in_pde_conv= REBNCONV(in_ch, out_ch=1, dirate=1)
        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv1 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=2)



        self.rebnconv3d = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv2d = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.fuse3 = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.fuse2 = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.fuse1 = REBNCONV(mid_ch*2,out_ch,dirate=1)


        self.fcl1 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl2 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl3 = nn.Conv2d(mid_ch, 1, 1)
        self.fcl3f = nn.Conv2d(mid_ch, 1, 1)
        self.fcl2f = nn.Conv2d(mid_ch, 1, 1)
        self.fcl1f = nn.Conv2d(out_ch, 1, 1)

        self.tfd1 = pde()
        self.tfd2 = pde()
        self.tfd3 = pde()
        self.tfd3f = pde()
        self.tfd2f = pde()
        self.tfd1f = pde()

        self.out_conv = REBNCONV(out_ch, out_ch, 1)

    def forward(self,x):
        _, _, hei, wid = x.shape
        hx = x

        x_p = self.in_pde_conv(hx)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)


        # ------------decoder-----------

        hx3d = self.rebnconv3d(hx)
        hx3dup = _upsample_like(hx3d, hx3)
        hx3f = self.fuse3(torch.cat((hx3dup, hx3),1))

        hx2d = self.rebnconv2d(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2),1))

        hx1d = self.rebnconv1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1),1))


        gf1 = self.fcl1(hx1)
        gf2 = F.interpolate(self.fcl2(hx2), size=[hei, wid], mode='bilinear', align_corners=True)
        gf3 = F.interpolate(self.fcl3(hx3), size=[hei, wid], mode='bilinear', align_corners=True)
        gf3f = F.interpolate(self.fcl3f(hx3f), size=[hei, wid], mode='bilinear', align_corners=True)
        gf2f = F.interpolate(self.fcl2f(hx2f), size=[hei, wid], mode='bilinear', align_corners=True)
        gf1f = self.fcl1f(hx1f)

        s1 = self.tfd1(x_p, gf1)
        s2 = self.tfd2(s1, gf2)
        s3 = self.tfd3(s2, gf3)
        s3f = self.tfd3f(s3, gf3f)
        s2f = self.tfd2f(s3f, gf2f)
        s1f = self.tfd1f(s2f, gf1f)

        pde_pred = s1f

        # Out

        hx_out = hx1f + hx1f * pde_pred

        out = self.out_conv(hx_out)

        return out

##### UIU-net ####
class PIP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, mode='train', deepsuper=True):
        super(PIP, self).__init__()
        self.mode = mode
        self.deepsuper = deepsuper

        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.pool2 = nn.MaxPool2d(4,stride=4,ceil_mode=True)
        self.pool3 = nn.MaxPool2d(8,stride=8,ceil_mode=True)
        self.pool4 = nn.MaxPool2d(16,stride=16,ceil_mode=True)

        self.stage1 = PDEU3(in_ch, 32, 64)
        self.stage2 = PDEU2(64,32,128)
        self.stage3 = PDEU2(128,64,256)
        self.stage4 = PDEU3d(256,128,512)
        

        self.stage4d = PDEU3d(512,256,512)
        self.stage3d = PDEU2(512,128,256)
        self.stage2d = PDEU2(256,64,128)
        self.stage1d = PDEU3(128,32,64)


        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')
        self.fuse1 = self._fuse_layer(64, 64, 64, fuse_mode='AsymBi')

        #------------------------PDE--------------------------
        self.fcl1 = nn.Conv2d(64, 1, 1)
        self.fcl2 = nn.Conv2d(128, 1, 1)
        self.fcl3 = nn.Conv2d(256, 1, 1)
        self.fcl4 = nn.Conv2d(512, 1, 1)  
        self.fcl4f = nn.Conv2d(512, 1, 1)
        self.fcl3f = nn.Conv2d(256, 1, 1)
        self.fcl2f = nn.Conv2d(128, 1, 1)
        self.fcl1f = nn.Conv2d(64, 1, 1)

        self.side1 = nn.Conv2d(512, out_ch, 1)
        self.side2 = nn.Conv2d(512, out_ch, 1)
        self.side3 = nn.Conv2d(256, out_ch, 1)
        self.side4 = nn.Conv2d(128, out_ch, 1)
        self.side5 = nn.Conv2d(64, out_ch, 1)

        self.tfd1 = pde()
        self.tfd2 = pde()
        self.tfd3 = pde()
        self.tfd4 = pde()
        self.tfd4f = pde()
        self.tfd3f = pde()
        self.tfd2f = pde()
        self.tfd1f = pde()

        self.in_pde_conv = nn.Conv2d(in_ch, 1, 3, 1, 1)



        self.out_conv = nn.Conv2d(64, out_ch, 1)
        self.outconv = nn.Conv2d(5 * out_ch, out_ch, 1)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels,fuse_mode='AsymBi'):#fuse_mode='AsymBi'

        if fuse_mode == 'AsymBi':
            fuse_layer = Fuse(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer


    def forward(self, x):
        _, _, hei, wid = x.shape
        hx = x
        x_p = self.in_pde_conv(x)

        #stage 1
        hx1 = self.stage1(hx)
        gf1 = self.fcl1(hx1)
        s1 = self.tfd1(x_p,gf1)
        m1 =s1
        hx1 = hx1*m1+hx1
        hx= self.pool(hx1)

        #stage 2
        hx2,s2_1,s2_2,s2_2d,s2_1d = self.stage2(hx)
        gf2 = F.interpolate(self.fcl2(hx2), size=[hei, wid], mode='bilinear', align_corners=True)
        s2 = self.tfd2(s1, gf2)
        m2 = self.pool(s2)
        hx2 = hx2*m2+hx2
        hx = self.pool(hx2)

        #stage 3
        hx3, _, _, _, _ = self.stage3(hx)
        gf3 = F.interpolate(self.fcl3(hx3), size=[hei, wid], mode='bilinear', align_corners=True)
        s3 = self.tfd3(s2, gf3)
        m3 = self.pool2(s3)
        hx3 = hx3*m3+hx3
        hx = self.pool(hx3)

        # stage 4

        hx4 = self.stage4(hx)
        gf4 = F.interpolate(self.fcl4(hx4), size=[hei, wid], mode='bilinear', align_corners=True)
        s4 = self.tfd4(s3, gf4)
        m4 = self.pool3(s4)
        hx4 = hx4*m4+hx4
        hx = self.pool(hx4)

        #-------------------- decoder --------------------

        hx4d = self.stage4d(hx)
        hx4dup = _upsample_like(hx4d,hx4)
        hx4f = self.fuse4(hx4dup, hx4)  
        gf4f = F.interpolate(self.fcl4f(hx4f), size=[hei, wid], mode='bilinear', align_corners=True)
        s4f = self.tfd4f(s4, gf4f)
        m4f = self.pool3(s4f)
        hx4f = hx4f*m4f+hx4f

        hx3d, _, _, _, _ = self.stage3d(hx4f)
        hx3dup = _upsample_like(hx3d,hx3)
        hx3f = self.fuse3(hx3dup, hx3)
        gf3f = F.interpolate(self.fcl3f(hx3f), size=[hei, wid], mode='bilinear', align_corners=True)
        s3f = self.tfd3f(s4f, gf3f)
        m3f = self.pool2(s3f)
        hx3f = hx3f*m3f+hx3f

        hx2d,_, _, _, _ = self.stage2d(hx3f)
        hx2dup = _upsample_like(hx2d,hx2)
        hx2f = self.fuse2(hx2dup,hx2)
        gf2f = F.interpolate(self.fcl2f(hx2f), size=[hei, wid], mode='bilinear', align_corners=True)
        s2f = self.tfd2f(s3f, gf2f)
        m2f = self.pool(s2f)
        hx2f = hx2f*m2f+hx2f

        hx1d = self.stage1d(hx2f)
        hx1dup = _upsample_like(hx1d,hx1)
        hx1f = self.fuse1(hx1dup,hx1)
        gf1f = self.fcl1f(hx1f)
        s1f = self.tfd1f(s2f, gf1f)
        m1f = s1f
        hx1f = hx1f*m1f+hx1f

        #--------------------deep supervision-------------------
        if self.deepsuper:
            d5 = F.interpolate(self.side1(hx4), size=[hei, wid])
            d4 = F.interpolate(self.side2(hx4f), size=[hei, wid])
            d3 = F.interpolate(self.side3(hx3f), size=[hei, wid])
            d2 = F.interpolate(self.side4(hx2f), size=[hei, wid])
            d1 = self.side5(hx1f)
            out = self.outconv(torch.cat((d1,d2,d3,d4,d5),1))      
            if self.mode == 'train':

                return torch.sigmoid(out)
                
                    
            else:

                return torch.sigmoid(out), m1, m2, m3, m4, m4f, m3f, m2f, m1f, s2_1, s2_2, s2_2d, s2_1d
        else:
            
            return torch.sigmoid(out = self.out_conv(hx1f))
        


if __name__ =='__main__':

    model = PIP(1,1,mode='train', deepsuper=True)
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')