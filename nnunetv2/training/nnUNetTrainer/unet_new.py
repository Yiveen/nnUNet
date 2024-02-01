from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from .ExtendDecoder import ExtendUNetDecoder

class ExtendConvUNet(PlainConvUNet):
    def __init__(self,
                 input_channels: int, #1
                 n_stages: int, # 6
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd], #3d
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int, #2
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False):
        '''
        {'conv_bias': True, 'norm_op': <class 'torch.nn.modules.instancenorm.InstanceNorm3d'>, 'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': <class 'torch.nn.modules.activation.LeakyReLU'>, 'nonlin_kwargs': {'inplace': True}}

        '''
        super(ExtendConvUNet, self).__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                                             conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.decoder = ExtendUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        seg, key = self.decoder(skips)
        return seg, key

