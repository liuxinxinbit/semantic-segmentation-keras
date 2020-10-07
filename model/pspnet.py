import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,\
    LeakyReLU,AveragePooling2D,DepthwiseConv2D,ZeroPadding2D, Add,UpSampling2D,Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import multi_gpu_model
import os
import random
import numpy as np
from PIL import Image
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
from math import ceil
from model.net_parts import build_conv2D_block, build_conv2Dtranspose_block,bottleneck,pyramid_pooling,build_SeparableConv2D_block,build_DepthwiseConv2D_block

def BN():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)

def residual_conv(prev, level, modify_stride=False):
    if modify_stride is False:
        prev = build_conv2D_block(prev, filters=64 * level, kernel_size=1, strides=1)
    elif modify_stride is True:
        prev = build_conv2D_block(prev, filters=64 * level, kernel_size=1, strides=2)
    prev = build_conv2D_block(prev, filters=64 * level, kernel_size=3, strides=1,dilation_rate=(1, 1))
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), use_bias=False)(prev)
    prev = BN()(prev)
    return prev


def short_convolution_branch(prev, level, modify_stride=False):

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1),
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2),
                      use_bias=False)(prev)
    prev = BN()(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

def residual_empty(prev_layer, level):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added

def Interp(x, shape):
    ''' 对图片做一个放缩，配合Keras的Lambda层使用'''
    # from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = tf.image.resize(x, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    return resized

class pspnet:
    def __init__(self,  print_summary=False,image_size=(512, 512, 3),num_class=3):
        self.num_class = num_class
        self.build(print_summary=print_summary,image_size=image_size)
        self.batch_generator =  None
        
    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='pspnet_model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='pspnet_model.h5'):
        self.model.load_weights(file_path)
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        self.model.fit(self.batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
    def ResNet(self, inputdata, layers):
        cnv1 = build_conv2D_block(inputdata, filters=64, kernel_size=3, strides=2)
        cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
        cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
        res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(cnv1)

        # 2_1- 2_3
        res = residual_short(res, 1)
        for i in range(2):
            res = residual_empty(res, 1)
        # 3_1 - 3_3
        res = residual_short(res, 2, modify_stride=True)
        for i in range(3):
            res = residual_empty(res, 2)
        if layers == 50:
            # 4_1 - 4_6
            res = residual_short(res, 4)
            for i in range(5):
                res = residual_empty(res, 4)
        elif layers == 101:
            # 4_1 - 4_23
            res = residual_short(res, 4)
            for i in range(22):
                res = residual_empty(res, 4)
        else:
            print("This ResNet is not implemented")

        # 5_1 - 5_3
        res = residual_short(res, 8)
        for i in range(2):
            res = residual_empty(res, 8)
        res = Activation('relu')(res)
        return res

    def interp_block(self, prev_layer, level, feature_map_shape, input_shape):
        if input_shape == (input_shape[0], input_shape[1]):
            kernel_strides_map = {1: 64,
                                  2: 32,
                                  4: 16,
                                  8: 8}
        else:
            print("Pooling parameters for input shape ",
                input_shape, " are not defined.")
            exit(1)
        kernel = (kernel_strides_map[level], kernel_strides_map[level])
        strides = (kernel_strides_map[level], kernel_strides_map[level])
        prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
        prev_layer = Conv2D(512, (1, 1), strides=(1, 1),use_bias=False)(prev_layer)
        prev_layer = BN()(prev_layer)
        prev_layer = Activation('relu')(prev_layer)
        prev_layer = UpSampling2D(size=(int(feature_map_shape[0]/level),int(feature_map_shape[1]/level)))(prev_layer)
        # prev_layer = Interp(feature_map_shape)(prev_layer)
        return prev_layer

    def build_pyramid_pooling_module(self, res, input_shape):
        """Build the Pyramid Pooling Module."""
        # ---PSPNet concat layers with Interpolation
        feature_map_size = tuple(int(ceil(input_dim / 8.0))
                                 for input_dim in input_shape)
        print("PSP module will interpolate to a final feature map size of %s" %
            (feature_map_size, ))

        interp_block1 = self.interp_block(res, 1, feature_map_size, input_shape)
        interp_block2 = self.interp_block(res, 2, feature_map_size, input_shape)
        interp_block3 = self.interp_block(res, 4, feature_map_size, input_shape)
        interp_block6 = self.interp_block(res, 8, feature_map_size, input_shape)

        res = concatenate([res,
                        interp_block6,
                        interp_block3,
                        interp_block2,
                        interp_block1],-1)
        return res


    def build(self, print_summary=False,image_size=(512, 512, 3), num_classes=3,resnet_layers=101):
        inputdata = Input(shape=image_size)
        res = self.ResNet(inputdata, layers=resnet_layers)
        psp = self.build_pyramid_pooling_module(res, (image_size[0],image_size[1]))
        x = build_conv2D_block(psp,512, (3, 3), strides=(1, 1),use_bias=False)
        x = Dropout(0.1)(x)
        x = Conv2D(num_classes, (1, 1), strides=(1, 1))(x)
        x = Lambda(Interp, arguments={'shape': (image_size[0], image_size[1])})(x) # 使用Lambda层放缩到原图片大小

        x = Activation('softmax')(x)            
        self.model = Model(inputs=inputdata, outputs=x)
        if print_summary:
            print(self.model.summary())
        # # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        # Solver
        sgd = SGD(lr=1e-3 , momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# PSPNet = pspnet(image_size = (512, 512, 3),num_class=3,print_summary=True)
