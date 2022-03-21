# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
from optical_flow_warp_fwd import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_samplerzb import *

monodepth_parameters = namedtuple('parameters',
                                  'height, width, '
                                  'batch_size, '
                                  'num_threads, '
                                  'num_epochs, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'dp_consistency_sigmoid_scale, '
                                  'disp_gradient_loss_weight, '
                                  'centerSymmetry_loss_weight, '
                                  'disp_consistency_loss_weight, '
                                  'full_summary')

"""monodepth LF model"""


class MonodepthModel(object):

    def __init__(self, params, mode, images_list, reuse_variables=None, model_index=None):
        self.params = params
        self.mode = mode
        self.model_collection = ['model_' + str(model_index)]

        self.images_list = images_list
        self.center = self.images_list[0]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def generate_image_top(self, img, disp):
        return bilinear_sampler_1d_v(img, -disp)

    def generate_image_bottom(self, img, disp):
        return bilinear_sampler_1d_v(img, disp)

    def generate_image_topleft(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, -disp_x, -disp_y)

    def generate_image_topright(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, disp_x, -disp_y)

    def generate_image_bottomleft(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, -disp_x, disp_y)

    def generate_image_bottomright(self, img, disp_x, disp_y):
        return bilinear_sampler_2d(img, disp_x, disp_y)

    ########################create by zec
    def generate_image_left_zec(self, img, disp, times):
        # return bilinear_sampler_1d_h(img, -disp)
        return bilinear_sampler_1d_h(img, -disp * times)

    def generate_image_right_zec(self, img, disp, times):
        # return bilinear_sampler_1d_h(img, disp)
        return bilinear_sampler_1d_h(img, disp * times)

    def generate_image_top_zec(self, img, disp, times):
        return bilinear_sampler_1d_v(img, -disp * times)

    def generate_image_topLeft_zec(self, img, disp, times):
        medile_result = bilinear_sampler_1d_h(img, -disp * times)
        return bilinear_sampler_1d_v(medile_result, -disp * times)

    def generate_image_topRight_zec(self, img, disp, times):
        medile_result = bilinear_sampler_1d_h(img, disp * times)
        return bilinear_sampler_1d_v(medile_result, -disp * times)

    def generate_image_bottom_zec(self, img, disp, times):
        return bilinear_sampler_1d_v(img, disp * times)

    def generate_image_bottomLeft_zec(self, img, disp, times):
        medile_result = bilinear_sampler_1d_h(img, -disp * times)
        return bilinear_sampler_1d_v(medile_result, disp * times)

    def generate_image_bottomRight_zec(self, img, disp, times):
        medile_result = bilinear_sampler_1d_h(img, disp * times)
        return bilinear_sampler_1d_v(medile_result, disp * times)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = tf.abs(disp_gradients_x * weights_x)
        smoothness_y = tf.abs(disp_gradients_y * weights_y)
        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    # it's checked with geonet
    def get_disp(self, x):
        # 9 out, the range of sigmoid is (0,1), however, the range of disparity is (-4,4)
        # disp = (self.conv(x, 9, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * 8
        disp = (self.conv(x, 9, 3, 1, activation_fn=tf.nn.sigmoid) - 0.5) * 8
        return disp

    """
    # it's checked with geonet
    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)
    """

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    # it's checked with geonet
    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    # it's checked with geonet
    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    # it's checked with geonet
    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    # it's checked with geonet
    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            # concat1 = tf.concat([upconv1, self.center, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.model_input = self.center
                # build model
                self.build_resnet50()

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            # 9 out
            self.disp_topleft_est = tf.expand_dims(self.disp1[:, :, :, 0], 3)
            self.disp_top_est = tf.expand_dims(self.disp1[:, :, :, 1], 3)
            self.disp_topright_est = tf.expand_dims(self.disp1[:, :, :, 2], 3)
            self.disp_left_est = tf.expand_dims(self.disp1[:, :, :, 3], 3)
            self.disp_center_est = tf.expand_dims(self.disp1[:, :, :, 4], 3)
            self.disp_right_est = tf.expand_dims(self.disp1[:, :, :, 5], 3)
            self.disp_bottomleft_est = tf.expand_dims(self.disp1[:, :, :, 6], 3)
            self.disp_bottom_est = tf.expand_dims(self.disp1[:, :, :, 7], 3)
            self.disp_bottomright_est = tf.expand_dims(self.disp1[:, :, :, 8], 3)

            self.disp_est_list = [tf.expand_dims(self.disp1[:, :, :, i], 3) for i in range(9)]

