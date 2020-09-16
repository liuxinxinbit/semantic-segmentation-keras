import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,\
    LeakyReLU,AveragePooling2D,DepthwiseConv2D,ZeroPadding2D, Add
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
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
from net_parts import build_conv2D_block, build_conv2Dtranspose_block,bottleneck,pyramid_pooling,build_SeparableConv2D_block,build_DepthwiseConv2D_block

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



    def build(self, print_summary=False,image_size=(512, 512, 3), num_classes=3,resnet_layers=50):
        inputdata = Input(shape=image_size)
        res = self.ResNet(inputdata, layers=resnet_layers)

        # output = Conv2DTranspose(filters=self.num_class, kernel_size=1, strides=1, activation='softmax', padding='same', name='output')(conv2d_deconv0)
            
        self.model = Model(inputs=inputdata, outputs=res)
        if print_summary:
            print(self.model.summary())
        # # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        # self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
PSPNet = pspnet(image_size = (512, 512, 3),num_class=3,print_summary=True)
