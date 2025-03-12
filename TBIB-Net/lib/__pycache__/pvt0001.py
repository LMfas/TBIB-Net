import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import pywt
import torch.nn.functional as F

# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)

# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())

#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# class ASPP(nn.Module):
#     def __init__(self, in_channels, atrous_rates, out_channels=64):
#         super(ASPP, self).__init__()
#         modules = []
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))

#         rates = tuple(atrous_rates)
#         for rate in rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))

#         modules.append(ASPPPooling(in_channels, out_channels))

#         self.convs = nn.ModuleList(modules)

#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))

#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)





# class DSC(object):

#     def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
#         self.num_points = kernel_size
#         self.width = input_shape[2]
#         self.height = input_shape[3]
#         self.morph = morph
#         self.device = device
#         self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

#         # define feature map shape
#         """
#         B: Batch size  C: Channel  W: Width  H: Height
#         """
#         self.num_batch = input_shape[0]
#         self.num_channels = input_shape[1]

#     """
#     input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
#     output_x: [B,1,W,K*H]   coordinate map
#     output_y: [B,1,K*W,H]   coordinate map
#     """

#     def _coordinate_map_3D(self, offset, if_offset):
#         # offset
#         y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

#         y_center = torch.arange(0, self.width).repeat([self.height])
#         y_center = y_center.reshape(self.height, self.width)
#         y_center = y_center.permute(1, 0)
#         y_center = y_center.reshape([-1, self.width, self.height])
#         y_center = y_center.repeat([self.num_points, 1, 1]).float()
#         y_center = y_center.unsqueeze(0)

#         x_center = torch.arange(0, self.height).repeat([self.width])
#         x_center = x_center.reshape(self.width, self.height)
#         x_center = x_center.permute(0, 1)
#         x_center = x_center.reshape([-1, self.width, self.height])
#         x_center = x_center.repeat([self.num_points, 1, 1]).float()
#         x_center = x_center.unsqueeze(0)

#         if self.morph == 0:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: only need 0
#                 x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
#             """
#             y = torch.linspace(0, 0, 1)
#             x = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )

#             y, x = torch.meshgrid(y, x)
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

#             y_offset_new = y_offset.detach().clone()

#             if if_offset:
#                 y_offset = y_offset.permute(1, 0, 2, 3)
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)

#                 # The center position remains unchanged and the rest of the positions begin to swing
#                 # This part is quite simple. The main idea is that "offset is an iterative process"
#                 y_offset_new[center] = 0
#                 for index in range(1, center):
#                     y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
#                     y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
#                 y_new = y_new.add(y_offset_new.mul(self.extend_scope))

#             y_new = y_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             return y_new, x_new

#         else:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 x: only need 0
#             """
#             y = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )
#             x = torch.linspace(0, 0, 1)

#             y, x = torch.meshgrid(y, x)
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1)

#             y_new = y_new.to(self.device)
#             x_new = x_new.to(self.device)
#             x_offset_new = x_offset.detach().clone()

#             if if_offset:
#                 x_offset = x_offset.permute(1, 0, 2, 3)
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)
#                 x_offset_new[center] = 0
#                 for index in range(1, center):
#                     x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
#                     x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
#                 x_new = x_new.add(x_offset_new.mul(self.extend_scope))

#             y_new = y_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             return y_new, x_new

#     """
#     input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
#     output: [N,1,K*D,K*W,K*H]  deformed feature map
#     """

#     def _bilinear_interpolate_3D(self, input_feature, y, x):
#         y = y.reshape([-1]).float()
#         x = x.reshape([-1]).float()

#         zero = torch.zeros([]).int()
#         max_y = self.width - 1
#         max_x = self.height - 1

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y)
#         y1 = torch.clamp(y1, zero, max_y)
#         x0 = torch.clamp(x0, zero, max_x)
#         x1 = torch.clamp(x1, zero, max_x)

#         input_feature_flat = input_feature.flatten()
#         input_feature_flat = input_feature_flat.reshape(
#             self.num_batch, self.num_channels, self.width, self.height)
#         input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
#         input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
#         dimension = self.height * self.width

#         base = torch.arange(self.num_batch) * dimension
#         base = base.reshape([-1, 1]).float()

#         repeat = torch.ones([self.num_points * self.width * self.height
#                              ]).unsqueeze(0)
#         repeat = repeat.float()

#         base = torch.matmul(base, repeat)
#         base = base.reshape([-1])

#         base = base.to(self.device)

#         base_y0 = base + y0 * self.height
#         base_y1 = base + y1 * self.height

#         # top rectangle of the neighbourhood volume
#         index_a0 = base_y0 - base + x0
#         index_c0 = base_y0 - base + x1

#         # bottom rectangle of the neighbourhood volume
#         index_a1 = base_y1 - base + x0
#         index_c1 = base_y1 - base + x1

#         # get 8 grid values
#         value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
#         value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
#         value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
#         value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y + 1)
#         y1 = torch.clamp(y1, zero, max_y + 1)
#         x0 = torch.clamp(x0, zero, max_x + 1)
#         x1 = torch.clamp(x1, zero, max_x + 1)

#         x0_float = x0.float()
#         x1_float = x1.float()
#         y0_float = y0.float()
#         y1_float = y1.float()

#         vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
#         vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
#         vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
#         vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

#         outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
#                    value_c1 * vol_c1)

#         if self.morph == 0:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 self.num_points * self.width,
#                 1 * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         else:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 1 * self.width,
#                 self.num_points * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         return outputs

#     def deform_conv(self, input, offset, if_offset):
#         y, x = self._coordinate_map_3D(offset, if_offset)
#         deformed_feature = self._bilinear_interpolate_3D(input, y, x)
#         return deformed_feature


# class DSConv(nn.Module):

#     def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
#                  if_offset, device = "cuda"):
#         """
#         The Dynamic Snake Convolution
#         :param in_ch: input channel
#         :param out_ch: output channel
#         :param kernel_size: the size of kernel
#         :param extend_scope: the range to expand (default 1 for this method)
#         :param morph: the morphology of the convolution kernel is mainly divided into two types
#                         along the x-axis (0) and the y-axis (1) (see the paper for details)
#         :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
#         :param device: set on gpu
#         """
#         super(DSConv, self).__init__()
#         # use the <offset_conv> to learn the deformable offset
#         self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
#         self.bn = nn.BatchNorm2d(2 * kernel_size)
#         self.kernel_size = kernel_size

#         # two types of the DSConv (along x-axis and y-axis)
#         self.dsc_conv_x = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(kernel_size, 1),
#             stride=(kernel_size, 1),
#             padding=0,
#         )
#         self.dsc_conv_y = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(1, kernel_size),
#             stride=(1, kernel_size),
#             padding=0,
#         )

#         self.gn = nn.GroupNorm(out_ch // 4, out_ch)
#         self.relu = nn.ReLU(inplace=True)

#         self.extend_scope = extend_scope
#         self.morph = morph
#         self.if_offset = if_offset
#         self.device = device

#     def forward(self, f):
#         offset = self.offset_conv(f)
#         offset = self.bn(offset)
#         # We need a range of deformation between -1 and 1 to mimic the snake's swing
#         offset = torch.tanh(offset)
#         input_shape = f.shape
#         dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
#                   self.device)
#         deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
#         if self.morph == 0:
#             x = self.dsc_conv_x(deformed_feature)
#             x = self.gn(x)
#             x = self.relu(x)
#             return x
#         else:
#             x = self.dsc_conv_y(deformed_feature)
#             x = self.gn(x)
#             x = self.relu(x)
#             return x


# # class _AtrousSpatialPyramidPoolingModule(nn.Module):
# #     """
# #     operations performed:
# #       1x1 x depth
# #       3x3 x depth dilation 6
# #       3x3 x depth dilation 12
# #       3x3 x depth dilation 18
# #       image pooling
# #       concatenate all together
# #       Final 1x1 conv
# #     """

# #     def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
# #         super(_AtrousSpatialPyramidPoolingModule, self).__init__()

# #         if output_stride == 8:
# #             rates = [2 * r for r in rates]
# #         elif output_stride == 16:
# #             pass
# #         else:
# #             raise 'output stride of {} not supported'.format(output_stride)

# #         self.features = []
# #         # 1x1
# #         self.features.append(
# #             nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
# #                           Norm2d(reduction_dim), nn.ReLU(inplace=True)))
# #         # other rates
# #         for r in rates:
# #             self.features.append(nn.Sequential(
# #                 nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
# #                           dilation=r, padding=r, bias=False),
# #                 Norm2d(reduction_dim),
# #                 nn.ReLU(inplace=True)
# #             ))
# #         self.features = torch.nn.ModuleList(self.features)

# #         # img level features
# #         self.img_pooling = nn.AdaptiveAvgPool2d(1)
# #         self.img_conv = nn.Sequential(
# #             nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
# #             Norm2d(reduction_dim), nn.ReLU(inplace=True))

# #     def forward(self, x):
# #         x_size = x.size()

# #         img_features = self.img_pooling(x)
# #         img_features = self.img_conv(img_features)
# #         img_features = Upsample(img_features, x_size[2:])
# #         out = img_features

# #         for f in self.features:
# #             y = f(x)
# #             out = torch.cat((out, y), 1)
# #         return out




# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
            
#             nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            
#             nn.BatchNorm2d(F_int)
#             )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)

#         return x * psi



# BatchNorm2d = nn.BatchNorm2d
# bn_mom = 0.1
# # 修改
# class DAPPM(nn.Module):
#     def __init__(self, inplanes, branch_planes, outplanes):
#         super(DAPPM, self).__init__()
#         self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                     BatchNorm2d(inplanes, momentum=bn_mom),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#                                     )
#         self.scale0 = nn.Sequential(
#             BatchNorm2d(inplanes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
#         )
#         self.process1 = nn.Sequential(
#             BatchNorm2d(branch_planes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#         )
#         self.process2 = nn.Sequential(
#             BatchNorm2d(branch_planes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#         )
#         self.process3 = nn.Sequential(
#             BatchNorm2d(branch_planes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#         )
#         self.process4 = nn.Sequential(
#             BatchNorm2d(branch_planes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
#         )
#         self.compression = nn.Sequential(
#             BatchNorm2d(branch_planes * 5, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
#         )
#         self.shortcut = nn.Sequential(
#             BatchNorm2d(inplanes, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
#         )

#     def forward(self, x):
#         # x = self.downsample(x)
#         width = x.shape[-1]
#         height = x.shape[-2]
#         x_list = []

#         x_list.append(self.scale0(x))
#         x_list.append(self.process1((F.interpolate(self.scale1(x),
#                                                    size=[height, width],
#                                                    mode='bilinear') + x_list[0])))
#         x_list.append((self.process2((F.interpolate(self.scale2(x),
#                                                     size=[height, width],
#                                                     mode='bilinear') + x_list[1]))))
#         x_list.append(self.process3((F.interpolate(self.scale3(x),
#                                                    size=[height, width],
#                                                    mode='bilinear') + x_list[2])))
#         x_list.append(self.process4((F.interpolate(self.scale4(x),
#                                                    size=[height, width],
#                                                    mode='bilinear') + x_list[3])))

#         out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
#         return out


# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()

#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x

# class BCA(nn.Module):
#     def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
#         super(BCA, self).__init__()
#         self.mid_channels = mid_channels
#         self.f_self = nn.Sequential(
#             nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#             nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#         )
#         self.f_x = nn.Sequential(
#             nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#             nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#         )
#         self.f_y = nn.Sequential(
#             nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#             nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(mid_channels),
#         )
#         self.f_up = nn.Sequential(
#             nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             BatchNorm(xin_channels),
#         )
#         self.scale = scale
#         nn.init.constant_(self.f_up[1].weight, 0)
#         nn.init.constant_(self.f_up[1].bias, 0)

#     def forward(self, x, y):
#         batch_size = x.size(0)
#         fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
#         # print(fself.shape)
#         fself = fself.permute(0, 2, 1)
#         fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
#         # print(fx.shape)
#         fx = fx.permute(0, 2, 1)
#         fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
#         # print(fy.shape)
#         sim_map = torch.matmul(fx, fy)
#         if self.scale:
#             sim_map = (self.mid_channels ** -.5) * sim_map
#         sim_map_div_C = F.softmax(sim_map, dim=-1)
#         # print(sim_map_div_C.shape)
#         fout = torch.matmul(sim_map_div_C, fself)
#         fout = fout.permute(0, 2, 1).contiguous()
#         fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
#         out = self.f_up(fout)
#         return x + out

#     def generate_edge_conv(self, in_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, 256, 3),
#             self.BatchNorm(256),
#             nn.ReLU(inplace=True),
#         )





# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()

#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x


# class CFM(nn.Module):
#     def __init__(self, channel):
#         super(CFM, self).__init__()
#         self.relu = nn.ReLU(True)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

#         self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

#     def forward(self, x1, x2, x3):
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
#                * self.conv_upsample3(self.upsample(x2)) * x3

#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2 = self.conv_concat2(x2_2)

#         x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2 = self.conv_concat3(x3_2)

#         x1 = self.conv4(x3_2)

#         return x1




# class GCN(nn.Module):
#     def __init__(self, num_state, num_node, bias=False):
#         super(GCN, self).__init__()
#         self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

#     def forward(self, x):
#         h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
#         h = h - x
#         h = self.relu(self.conv2(h))
#         return h


# class SAM(nn.Module):
#     def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
#         super(SAM, self).__init__()

#         self.normalize = normalize
#         self.num_s = int(plane_mid)
#         self.num_n = (mids) * (mids)
#         self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

#         self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
#         self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
#         self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
#         self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

#     def forward(self, x, edge):
#         edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

#         n, c, h, w = x.size()
#         edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

#         x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
#         x_proj = self.conv_proj(x)
#         x_mask = x_proj * edge

#         x_anchor1 = self.priors(x_mask)
#         x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
#         x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

#         x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
#         x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

#         x_rproj_reshaped = x_proj_reshaped

#         x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
#         if self.normalize:
#             x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
#         x_n_rel = self.gcn(x_n_state)

#         x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
#         x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
#         out = x + (self.conv_extend(x_state))

#         return out


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


# class PolypPVT(nn.Module):
#     def __init__(self, channel=32):
#         super(PolypPVT, self).__init__()

#         self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
#         path = './pvt_v2_b2.pth'
#         save_model = torch.load(path)
#         model_dict = self.backbone.state_dict()
#         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         self.backbone.load_state_dict(model_dict)

#         self.Translayer2_0 = BasicConv2d(64, channel, 1)
#         self.Translayer2_1 = BasicConv2d(128, channel, 1)
#         self.Translayer3_1 = BasicConv2d(320, channel, 1)
#         self.Translayer4_1 = BasicConv2d(512, channel, 1)

# #  修改
#         # self.Translayer2_0 = BasicConv2d(64, 64, 1)
#         # self.Translayer2_1 = BasicConv2d(128, 64, 1)
#         # self.Translayer3_1 = BasicConv2d(320, 64, 1)
#         # self.Translayer4_1 = BasicConv2d(512, 64, 1)





#         self.CFM = CFM(channel)
#         self.ca = ChannelAttention(64)
#         self.sa = SpatialAttention()
#         self.SAM = SAM()
        
#         self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
#         self.out_SAM = nn.Conv2d(128, 1, 1)
#         self.out_CFM = nn.Conv2d(1, 1, 1)
#         self.out_CFM1 = nn.Conv2d(32, 1, 1)
#         # self.out_CFM1 = nn.Conv2d(32, 1, 1)

#         self.interpolate = F.interpolate


#         # self.DSConv = DSConv(64, 64, 3, 1, 0, True)

        
#         self.upsample_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)

#         self.edge_conv1 = self.generate_edge_conv(64)
#         self.edge_out1 = nn.Sequential(nn.Conv2d(64,1,1),
#                                        nn.Sigmoid())
#         self.edge_conv2 = self.generate_edge_conv(128)
#         self.edge_out2 = nn.Sequential(nn.Conv2d(64,1,1),
#                                        nn.Sigmoid())
#         self.edge_conv3 = self.generate_edge_conv(320)
#         self.edge_out3 = nn.Sequential(nn.Conv2d(64,1,1),
#                                        nn.Sigmoid())
#         self.edge_conv4 = self.generate_edge_conv(512)
#         self.edge_out4 = nn.Sequential(nn.Conv2d(64,1,1),
#                                        nn.Sigmoid())
        


#         self.att = BCA(64, 64, 64)
#         self.conv1 = nn.Conv2d(256, 256,kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv1bn = nn.BatchNorm2d(256)
#         self.conv1relu = nn.ReLU(inplace=True)



#         self.out_head1 = nn.Conv2d(512, 1, 1)
#         self.out_head2 = nn.Conv2d(320, 1, 1)
#         self.out_head3 = nn.Conv2d(128, 1, 1)
#         self.out_head4 = nn.Conv2d(64, 1, 1)






#         self.upsample_conv3 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )

#         self.upsample_conv4 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=320, out_channels=64, kernel_size=8, stride=4, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )


#         self.upsample_conv5 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=16, stride=8, padding=4),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )


#         self.upsample_conv6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=8, stride=4, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )


#         self.upsample_conv7 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )

#         self.jingxi = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )





#         self.down001 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2 , padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )


#         self.down002 = nn.Sequential(
#             nn.Conv2d(32, 128, kernel_size=3, stride=2 , padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
        
#         self.down004 = nn.Sequential(
#             nn.Conv2d(32, 512, kernel_size=4, stride=4 ,padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )

#         self.down005 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=4, stride=4 ,padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )

        
     





#         self.DAPPM = DAPPM(inplanes=512,branch_planes=512,outplanes=512)


#         self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


#         self.ASPP = ASPP(in_channels=64, atrous_rates=[6, 12, 18])



#         self.att4 = Attention_block(F_g=64,F_l=64,F_int=64)
        

#         self.down = nn.Sequential(
#             nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )


#     def generate_edge_conv(self, in_channels):
#          return nn.Sequential(
#             DSConv(in_channels, 64, 3, 1, 0, True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )






        




#     def forward(self, x):
#         x_size = x.size()

#         # backbone
#         pvt = self.backbone(x)
#         x1 = pvt[0]
#         x2 = pvt[1]
#         x3 = pvt[2]
#         x4 = pvt[3]



#         # x4_1 = self.DAPPM(x4)
#         x4_1 = x4 
        


#         # x1_t = self.Translayer2_0(x1)
#         x2_t = self.Translayer2_1(x2)  
#         x3_t = self.Translayer3_1(x3)  
#         x4_t = self.Translayer4_1(x4)  
#         # x4_t1 = self.Translayer4_1(x4_1) 



#         # p1 = self.upsample3(x4_t)
#         # p2 = self.upsample2(x3_t)
#         # p3 = self.upsample1(x2_t)
#         # p4 = self.upsample3(x4_t1)
#         # # p4 = self.out_head4(x1)

#         # P = p1 + p2 + p3 + p4 + x1_t 
#         # # P = p1 + p2 + p3 +  x1_t 




#         # CIM
#         x1 = self.ca(x1) * x1 # channel attention
#         cim_feature = self.sa(x1) * x1 # spatial attention


#         # CFM
#         x2_t = self.Translayer2_1(x2)  
#         x3_t = self.Translayer3_1(x3)  
#         x4_t = self.Translayer4_1(x4)  
#         cfm_feature = self.CFM(x4_t, x3_t, x2_t)
#         # cfm_feature = self.upsample_conv3(cfm_feature)

#         # print(cfm_feature.shape)
#         # print(cim_feature.shape)



#         # p01 = P 

#         SAM
#         T2 = self.Translayer2_0(cim_feature)
#         T2 = self.down05(T2)
#         sam_feature = self.SAM(cfm_feature, T2)



#         e1 = self.edge_conv1(x1)
#         e1 = self.interpolate(e1, x1.size()[2:], mode='bilinear', align_corners=True)
#         e1_out = self.edge_out1(e1)
#         # print(e1_out.shape)
#         e2 = self.edge_conv2(x2)
#         e2 = self.interpolate(e2, x1.size()[2:], mode='bilinear', align_corners=True)
#         e2_out = self.edge_out2(e2)
#         e3 = self.edge_conv3(x3)
#         e3 = self.interpolate(e3, x1.size()[2:], mode='bilinear', align_corners=True)
#         e3_out = self.edge_out3(e3)
#         e4 = self.edge_conv4(x4)
#         e4 = self.interpolate(e4, x1.size()[2:], mode='bilinear', align_corners=True)
#         e4_out = self.edge_out4(e4)
#         e = torch.cat((e1, e2, e3, e4), dim=1)
#         sam_feature_1 = self.down004(sam_feature)

#         x_br1 = sam_feature_1 + x4_1

#         x4_1_1 = self.upsample_conv6(x4_1)

#         x_br2 = x4_1_1 + sam_feature

#         x_br2_d = self.down002(x_br2)
#         x_br1_u = self.upsample_conv7(x_br1)

#         x_bb1 = x_br1_u * x_br2_d

#         e_1 = self.down005(e)

#         e_1_1 = e_1 * x_br1_u

#         out_feature = x_bb1 * e_1_1








       
#         # e10 = self.conv1(e)
#         # e10 = self.conv1bn(e10)
#         # e = self.conv1relu(e10)
#         # e10 = self.conv1(e10)
#         # e10 = self.conv1bn(e10)
#         # e10 = self.conv1relu(e10)
#         # e10 = self.conv1(e10)
#         # e10 = self.conv1bn(e10)
#         # e10 = self.conv1relu(e10)
#         # e = e + e10
#         # print(e.shape)
#         # e = e1 + e2 + e3 + e4
#         # e = self.down(e)
#         # print(e.shape)
#         bound_out_ = e1_out+e2_out+e3_out+e4_out
#         # print(bound_out_.shape)
        
#         # print(x4.shape)
#         # out_feature = self.att(p01, e)
#         # print(sam_feature.shape)
#         # print(e.shape)
#         # prediction4 = self.out_CFM1(sam_feature)
#         # sam_feature = self.upsample_conv3(sam_feature)
#         # print(e.shape)
#         # e = self.DSConv(e)
#         # e1 = self.edge_conv1(e)
#         # e = e + e1
#         # e2 = self.edge_conv1(e)
#         # e = e + e2
#         # e3 = self.edge_conv1(e)
#         # e = e + e3
#         # e4 = self.edge_conv1(e)
#         # e = e + e4
#         # e5 = self.edge_conv1(e)
#         # e = e + e5
#         # out_feature = sam_feature * e
#         # sam_feature1 = self.ASPP(sam_feature)
#         # sam_feature2 = sam_feature + sam_feature1
#         # e = self.ASPP(e)
#         # out_feature = sam_feature * e 
#         # out_feature = sam_feature2 + out_feature

#         # out_feature = self.ASPP(out_feature)
#         # out_feature = self.att4(sam_feature,e)
#         # out_feature = out_feature + out_feature1
#         # print(sam_feature.shape)
#         # print(e.shape)
#         # print(out_feature.shape)


#         prediction1 = self.out_CFM(bound_out_) 

#         prediction2 = self.out_SAM(out_feature)
#         # prediction3 = self.out_CFM1(p01)
#         prediction3 = self.out_CFM1(sam_feature)

#         # prediction4 = self.out_SAM(sam_feature)
        
#         # print(prediction1.shape)
#         # print(prediction2.shape)


#         # prediction1_8 = F.interpolate(prediction1, scale_factor=4, mode='bilinear') 
#         prediction2_8 = F.interpolate(prediction2, scale_factor=16, mode='bilinear') 
#         prediction1_8 = self.interpolate(prediction1, x_size[2:], mode='bilinear', align_corners=True)
#         prediction3_8 = F.interpolate(prediction3, scale_factor=8, mode='bilinear') 
#         # prediction4_8 = F.interpolate(prediction4, scale_factor=8, mode='bilinear') 
#         return prediction1_8, prediction2_8, prediction3_8

        
#         # CIM
#         # x1 = self.ca(x1) * x1 # channel attention
#         # cim_feature = self.sa(x1) * x1 # spatial attention


#         # # CFM
#         # x2_t = self.Translayer2_1(x2)  
#         # x3_t = self.Translayer3_1(x3)  
#         # x4_t = self.Translayer4_1(x4)  
#         # cfm_feature = self.CFM(x4_t, x3_t, x2_t)

#         # # SAM
#         # T2 = self.Translayer2_0(cim_feature)
#         # T2 = self.down05(T2)
#         # sam_feature = self.SAM(cfm_feature, T2)

#         # prediction1 = self.out_CFM(cfm_feature)
#         # prediction2 = self.out_SAM(sam_feature)

#         # prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
#         # prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
#         # return prediction1_8, prediction2_8


# if __name__ == '__main__':
#     model = PolypPVT().cuda()
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()

#     prediction1, prediction2, prediction3 = model(input_tensor)
#     print(prediction1.size(), prediction2.size(), prediction3.size())
# # 




















class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=64):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)





class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device = "cuda"):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# class _AtrousSpatialPyramidPoolingModule(nn.Module):
#     """
#     operations performed:
#       1x1 x depth
#       3x3 x depth dilation 6
#       3x3 x depth dilation 12
#       3x3 x depth dilation 18
#       image pooling
#       concatenate all together
#       Final 1x1 conv
#     """

#     def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
#         super(_AtrousSpatialPyramidPoolingModule, self).__init__()

#         if output_stride == 8:
#             rates = [2 * r for r in rates]
#         elif output_stride == 16:
#             pass
#         else:
#             raise 'output stride of {} not supported'.format(output_stride)

#         self.features = []
#         # 1x1
#         self.features.append(
#             nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#                           Norm2d(reduction_dim), nn.ReLU(inplace=True)))
#         # other rates
#         for r in rates:
#             self.features.append(nn.Sequential(
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
#                           dilation=r, padding=r, bias=False),
#                 Norm2d(reduction_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = torch.nn.ModuleList(self.features)

#         # img level features
#         self.img_pooling = nn.AdaptiveAvgPool2d(1)
#         self.img_conv = nn.Sequential(
#             nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#             Norm2d(reduction_dim), nn.ReLU(inplace=True))

#     def forward(self, x):
#         x_size = x.size()

#         img_features = self.img_pooling(x)
#         img_features = self.img_conv(img_features)
#         img_features = Upsample(img_features, x_size[2:])
#         out = img_features

#         for f in self.features:
#             y = f(x)
#             out = torch.cat((out, y), 1)
#         return out


class WaveletModule(nn.Module):
    def __init__(self, in_channels):
        super(WaveletModule, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.alpha = nn.Parameter(torch.ones(1))
        self.alpha = 0.5
        # self.beta = nn.Parameter(torch.ones(1))
        self.beta = 0.5
        self.bn = nn.BatchNorm2d(1)
        self.re = nn.ReLU()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 将模型移动到 GPU
        self.to(self.device)

    def forward(self, x):
       
        outputs = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 确保输入张量在相同的设备上
        x = x.to(device)

        # 对输入张量的每个通道进行处理
        for channel in range(self.in_channels):
            # 获取当前通道的数据
            channel_data = x[:, channel, :, :].unsqueeze(1)

            LLY, (LHY, HLY, HHY) = pywt.dwt2(channel_data.cpu().detach().numpy(), 'haar')





            
            LL = torch.tensor(LLY).to(self.device)
            
            

            LL1 = self.conv1(LL)
            LL1 = self.bn1(LL1)
            LL1 = self.relu1(LL1)
            LL1 = LL1 * self.alpha

            LL1 = LL + LL1

            LL2 = self.conv1(LL)
            LL2 = self.bn1(LL2)
            LL2 = self.relu1(LL2)
            
            LL2 = torch.cat((LL2, LL1), 1)  # 在通道维度上拼接
            
           

            LL2 = self.conv2(LL2)
            

            LL2 = 1 - LL2 * self.beta

            LL3 = LL1 + LL2









            LH = torch.tensor(LHY).to(device)


  
            
            

            LH1 = self.conv1(LH)
            LH1 = self.bn1(LH1)
            LH1 = self.relu1(LH1)
            LH1 = LH1 * self.alpha

            LH1 = LH + LH1

            LH2 = self.conv1(LH)
            LH2 = self.bn1(LH2)
            LH2 = self.relu1(LH2)
            
            LH2 = torch.cat((LH2, LH1), 1)  # 在通道维度上拼接
            
           

            LH2 = self.conv2(LH2)
            

            LH2 = 1 - LH2 * self.beta

            LH3 = LH1 + LH2

  
            HL = torch.tensor(HLY).to(device)

            HL1 = self.conv1(HL)
            HL1 = self.bn1(HL1)
            HL1 = self.relu1(HL1)
            HL1 = HL1 * self.alpha

            HL1 = HL + HL1

            HL2 = self.conv1(HL)
            HL2 = self.bn1(HL2)
            HL2 = self.relu1(HL2)
            HL2 = torch.cat((HL2, HL1), 1)  # 在通道维度上拼接
           

            HL2 = self.conv2(HL2)

            HL2 = 1 - HL2 * self.beta

            HL3 = HL1 + HL2


            HH = torch.tensor(HHY).to(device)

            HH1 = self.conv1(HH)
            HH1 = self.bn1(HH1)
            HH1 = self.relu1(HH1)
            HH1 = HH1 * self.alpha

            HH1 = HH + HH1

            HH2 = self.conv1(HH)
            HH2 = self.bn1(HH2)
            HH2 = self.relu1(HH2)
            HH2 = torch.cat((HH2, HH1), 1)  # 在通道维度上拼接
          

            HH2 = self.conv2(HH2)

            HH2 = 1 - HH2 * self.beta

            HH3 = HH1 + HH2

            LH4 = pywt.idwt2((LL3.cpu().detach().numpy(), (LH3.cpu().detach().numpy(),HL3.cpu().detach().numpy(),HH3.cpu().detach().numpy())), 'haar')
            LH4 = torch.tensor(LH4).to(device)
            

            outputs.append(LH4)

        # 将处理后的每个通道结果拼接起来
        result = torch.cat(outputs, 1)
       

        return result


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x * psi



BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
# 修改
class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BCA(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(BCA, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        # print(fself.shape)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        # print(fx.shape)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        # print(fy.shape)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        # print(sim_map_div_C.shape)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return x + out

    def generate_edge_conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3),
            self.BatchNorm(256),
            nn.ReLU(inplace=True),
        )





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1




class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PolypPVT(nn.Module):
    def __init__(self, channel=32):
        super(PolypPVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

#  修改
        # self.Translayer2_0 = BasicConv2d(64, 64, 1)
        # self.Translayer2_1 = BasicConv2d(128, 64, 1)
        # self.Translayer3_1 = BasicConv2d(320, 64, 1)
        # self.Translayer4_1 = BasicConv2d(512, 64, 1)





        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.ca2 = ChannelAttention(32)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(128, 1, 1)
        self.out_CFM = nn.Conv2d(1, 1, 1)
        self.out_CFM1 = nn.Conv2d(64, 1, 1)
        # self.out_CFM1 = nn.Conv2d(32, 1, 1)

        self.interpolate = F.interpolate


        # self.DSConv = DSConv(64, 64, 3, 1, 0, True)

        
        # self.upsample_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # self.upsample_conv4 = nn.ConvTranspose2d(in_channels=320, out_channels=64, kernel_size=8, stride=4, padding=2)
        # self.upsample_conv5 = nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=16, stride=8, padding=4)
        # self.upsample_conv6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=8, stride=4, padding=2)
        # self.upsample_conv7 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)


        self.upsample_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # self.upsample_conv8 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )

        self.upsample_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=320, out_channels=64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        self.upsample_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        self.upsample_conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.upsample_conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.jingxi = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )





        self.down001 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2 , padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.down002 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2 , padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )



        self.edge_conv1 = self.generate_edge_conv(64)
        self.edge_out1 = nn.Sequential(nn.Conv2d(64,1,1),
                                       nn.Sigmoid())
        self.edge_conv2 = self.generate_edge_conv(128)
        self.edge_out2 = nn.Sequential(nn.Conv2d(64,1,1),
                                       nn.Sigmoid())
        self.edge_conv3 = self.generate_edge_conv(320)
        self.edge_out3 = nn.Sequential(nn.Conv2d(64,1,1),
                                       nn.Sigmoid())
        self.edge_conv4 = self.generate_edge_conv(512)
        self.edge_out4 = nn.Sequential(nn.Conv2d(64,1,1),
                                       nn.Sigmoid())
        


        self.att = BCA(64, 64, 64)
        self.conv1 = nn.Conv2d(256, 256,kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1bn = nn.BatchNorm2d(256)
        self.conv2bn = nn.BatchNorm2d(512)
        self.conv1relu = nn.ReLU(inplace=True)



        self.out_head1 = nn.Conv2d(512, 1, 1)
        self.out_head2 = nn.Conv2d(320, 1, 1)
        self.out_head3 = nn.Conv2d(128, 1, 1)
        self.out_head4 = nn.Conv2d(64, 1, 1)


        self.DAPPM = DAPPM(inplanes=512,branch_planes=512,outplanes=512)


        self.wal1 = WaveletModule(64)
        self.wal2 = WaveletModule(128)
        self.wal3 = WaveletModule(320)





        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        self.ASPP = ASPP(in_channels=64, atrous_rates=[6, 12, 18])



        self.att4 = Attention_block(F_g=64,F_l=64,F_int=64)


        # self.conv_downsampling = nn.Conv2d(3,3,kernel_size=2,stride=2)
        self.max_pooling = nn.AvgPool2d(kernel_size=2)

        self.conv001 = nn.Conv2d(64,128,3,1,1)

        self.conv002 = nn.Conv2d(256,320,3,1,1)
        self.conv003 = nn.Conv2d(640,512,3,1,1)
        
        

        self.down = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        self.down004 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=8, stride=8 ,padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


    def generate_edge_conv(self, in_channels):
         return nn.Sequential(
            DSConv(in_channels, 64, 3, 1, 0, True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )







        




    def forward(self, x):
        x_size = x.size()

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]


        
        e1 = self.edge_conv1(x1)
        e1 = self.interpolate(e1, x1.size()[2:], mode='bilinear', align_corners=True)
        e1_out = self.edge_out1(e1)
        # print(e1_out.shape)
        e2 = self.edge_conv2(x2)
        e2 = self.interpolate(e2, x1.size()[2:], mode='bilinear', align_corners=True)
        e2_out = self.edge_out2(e2)
        e3 = self.edge_conv3(x3)
        e3 = self.interpolate(e3, x1.size()[2:], mode='bilinear', align_corners=True)
        e3_out = self.edge_out3(e3)
        e4 = self.edge_conv4(x4)
        e4 = self.interpolate(e4, x1.size()[2:], mode='bilinear', align_corners=True)
        e4_out = self.edge_out4(e4)
        e = torch.cat((e1, e2, e3, e4), dim=1)
        bound_out_ = e1_out+e2_out+e3_out+e4_out
        e = self.down002(e)
 

        # x1_pool = self.max_pooling(x1)
        # # print(x1_pool.shape)
        # x1_001 = self.conv001(x1_pool)
        # # print(x1_001.shape)
        # x0011_2 = torch.concat((x1_001,x2),dim=1)

        # x0011_2_pool = self.max_pooling(x0011_2)
        # x2_002 = self.conv002(x0011_2_pool)
        # x3_c = torch.concat((x2_002,x3),dim=1)
        # x3_c_pool = self.max_pooling(x3_c)
        # x3_ban = self.conv003(x3_c_pool)
        # x3_ban = self.conv2bn(x3_ban)
        # x3_ban = self.conv1relu(x3_ban)

        x4_1 = self.DAPPM(x4)
        x4_1 = x4 + x4_1


        x1 = self.ca(x1) * x1 # channel attention
        x1_cim = self.sa(x1) * x1 # spatial attention
        # x1_cim = x1 + x1_cim
        # # print(x1_cim.shape)


        # x1_cim1 = self.wal1(x1_cim)



        # x1_11 = x1_cim + x1_cim1
        # x2 = x21 + x2
        # x3 = x31 + x3



        x2 = self.upsample_conv3(x2)
        x3 = self.upsample_conv4(x3)
        x4 = self.upsample_conv5(x4)



        # x21 = self.wal1(x2)
        # x31 = self.wal1(x3)
        # x2 = x21 + x2
        # x3 = x31 + x3


        






 


        x_br1 = x1_cim + x2 + x3 + x4

        # print(x_br1.shape)
        # print(x4_1.shape)



        # x_br1_a = self.jingxi(x_br1)
        # x_br1_b = self.jingxi(x_br1_a)
        x_br1_1 = self.down004(x_br1)
        # print(x_br1.shape)

        x_br1_2 = x_br1_1 + x4_1

        # x_br1_1 = x_br1 + x_br1_b

        x4_1_3 = self.upsample_conv5(x4_1)

        x4_1_4 = x4_1_3 + x_br1



       

        x4_1_2 = self.upsample_conv6(x_br1_2)
        x1_dowm = self.down001(x4_1_4)
        x_1br1_sum = x4_1_2 * x1_dowm
        # # x_1br1_sum = torch.concat((x4_1_2,x1_dowm),dim = 1)
        x_br2 = x4_1_2 * e
        # # x_1br1_sum = self.upsample_conv8(x_1br1_sum)


        # out_feature = x1_dowm * x4_1_2


        out_feature = x_1br1_sum * x_br2


        






        # print(e.shape)


        # x_b_bruanch= e4_*x4_1









    






        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)



   



        # x_a_bruanch= x3_ban + x4_1

        # print(x_a_bruanch.shape)


        


        # x1_t = self.Translayer2_0(x1)
        # x2_t = self.Translayer2_1(x2)  
        # x3_t = self.Translayer3_1(x3)  
        # x4_t = self.Translayer4_1(x4)  
        # x4_t1 = self.Translayer4_1(x4_1) 



        # p1 = self.upsample3(x4_t)
        # p2 = self.upsample2(x3_t)
        # p3 = self.upsample1(x2_t)
        # p4 = self.upsample3(x4_t1)
        # # p4 = self.out_head4(x1)

        # P = p1 + p2 + p3 + p4 + x1_t 
        # # P = p1 + p2 + p3 +  x1_t 




        # CIM
        # x1 = self.ca(x1) * x1 # channel attention
        # cim_feature = self.sa(x1) * x1 # spatial attention


        # # CFM
        # x2_t = self.Translayer2_1(x2)  
        # x3_t = self.Translayer3_1(x3)  
        # x4_t = self.Translayer4_1(x4)  
        # cfm_feature = self.CFM(x4_t, x3_t, x2_t)
        # cfm_feature = self.upsample_conv3(cfm_feature)

        # print(cfm_feature.shape)
        # print(cim_feature.shape)



        # p01 = P 

        # SAM
        # T2 = self.Translayer2_0(cim_feature)
        # T2 = self.down05(T2)
        # sam_feature = self.SAM(cfm_feature, T2)






        # out_feature = x_a_bruanch + x_b_bruanch



        # e = torch.cat((e1, e2, e3, e4), dim=1)
        

        # print(e4_.shape)
       
        # e10 = self.conv1(e)
        # e10 = self.conv1bn(e10)
        # e = self.conv1relu(e10)
        # e10 = self.conv1(e10)
        # e10 = self.conv1bn(e10)
        # e10 = self.conv1relu(e10)
        # e10 = self.conv1(e10)
        # e10 = self.conv1bn(e10)
        # e10 = self.conv1relu(e10)
        # e = e + e10
        # print(e.shape)
        # e = e1 + e2 + e3 + e4
        # e = self.down(e)
        # print(e.shape)
        
        # print(bound_out_.shape)
        
        # print(x4.shape)
        # out_feature = self.att(p01, e)
        # print(sam_feature.shape)
        # print(e.shape)
        # prediction4 = self.out_CFM1(sam_feature)
        # sam_feature = self.upsample_conv3(sam_feature)
        # print(e.shape)
        # e = self.DSConv(e)
        # e1 = self.edge_conv1(e)
        # e = e + e1
        # e2 = self.edge_conv1(e)
        # e = e + e2
        # e3 = self.edge_conv1(e)
        # e = e + e3
        # e4 = self.edge_conv1(e)
        # e = e + e4
        # e5 = self.edge_conv1(e)
        # e = e + e5
        # out_feature = sam_feature * e
        # sam_feature1 = self.ASPP(sam_feature)
        # sam_feature2 = sam_feature + sam_feature1
        # e = self.ASPP(e)
        # out_feature = sam_feature * e 
        # out_feature = sam_feature2 + out_feature

        # out_feature = self.ASPP(out_feature)
        # out_feature = self.att4(sam_feature,e)
        # out_feature = out_feature + out_feature1
        # print(sam_feature.shape)
        # print(e.shape)
        # print(out_feature.shape)


        prediction1 = self.out_CFM(bound_out_) 

        prediction2 = self.out_SAM(out_feature)
        # prediction3 = self.out_CFM1(p01)
        prediction3 = self.out_CFM1(x_br1)

        # prediction4 = self.out_SAM(sam_feature)
        
        # print(prediction1.shape)
        # print(prediction2.shape)


        prediction1_8 = F.interpolate(prediction1, scale_factor=4, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear') 
        # prediction1_8 = self.interpolate(prediction1, x_size[2:], mode='bilinear', align_corners=True)
        prediction3_8 = F.interpolate(prediction3, scale_factor=4, mode='bilinear') 
        # prediction4_8 = F.interpolate(prediction4, scale_factor=8, mode='bilinear') 
        return  prediction1_8, prediction2_8, prediction3_8

        
        # CIM
        # x1 = self.ca(x1) * x1 # channel attention
        # cim_feature = self.sa(x1) * x1 # spatial attention


        # # CFM
        # x2_t = self.Translayer2_1(x2)  
        # x3_t = self.Translayer3_1(x3)  
        # x4_t = self.Translayer4_1(x4)  
        # cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # # SAM
        # T2 = self.Translayer2_0(cim_feature)
        # T2 = self.down05(T2)
        # sam_feature = self.SAM(cfm_feature, T2)

        # prediction1 = self.out_CFM(cfm_feature)
        # prediction2 = self.out_SAM(sam_feature)

        # prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        # prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        # return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1,prediction2,prediction3 = model(input_tensor)
    print(prediction1.size(),prediction2.size(),prediction3.size())
