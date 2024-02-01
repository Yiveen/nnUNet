import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from .attention import AttentionBlock

class ExtendUNetDecoder(UNetDecoder):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False, config_key_point: dict = None):
        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first)

        #Here is same as the seg branch decoder
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder
        # we start with the bottleneck and work out way up
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        stages_key = []
        transpconvs_key = []
        attention_key = []
        seg_layers_key = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs_key.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages_key.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))
            if s != n_stages_encoder - 1:
                attention_key.append(AttentionBlock(input_features_skip, input_features_skip, input_features_skip/2))

        self.stages_key = nn.ModuleList(stages_key)
        self.transpconvs_key = nn.ModuleList(transpconvs_key)
        self.attention_key = nn.ModuleList(attention_key)
    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        lkey_input = skips[-1]
        seg_outputs = []
        key_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x_key = self.stages_key[s](lkey_input)

            x = torch.cat((x, skips[-(s+2)]), 1)
            x_key = torch.cat((x_key, skips[-(s+2)]), 1)

            x = self.stages[s](x)
            x_key = self.stages_key[s](x_key)
            x_key = self.attention_key[s](x, x_key)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
                key_outputs.append(x_key)
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
                key_outputs.append(x_key)
            lres_input = x
            lkey_input = x_key

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
            k = key_outputs[0]
        else:
            r = seg_outputs
            k = key_outputs[-1]
        return r, k