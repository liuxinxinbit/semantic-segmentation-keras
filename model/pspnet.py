import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,LeakyReLU,AveragePooling2D,DepthwiseConv2D,ZeroPadding2D
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
from .net_parts import build_conv2D_block, build_conv2Dtranspose_block,bottleneck,pyramid_pooling,build_SeparableConv2D_block,build_DepthwiseConv2D_block

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

    def build(self, print_summary=False,image_size=(448, 512, 3), num_classes=3):
        inputdata = Input(shape=image_size)

        cnv1 = build_conv2D_block(inputdata, filters=64, kernel_size=3, strides=2)
        cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
        cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
        res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(cnv1)

        # 2_1- 2_3
        res = residual_short(res, 1, pad=1)
        for i in range(2):
            res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)




        # Global feature extractor

        global_feature_extractor1 = bottleneck(learning_to_down_sample3,
                                          64, 2, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor1,
                                          64, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 2, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = pyramid_pooling(global_feature_extractor,
                                               sub_region_sizes)

        # Feature fusion

        conv2d_deconv5_1 = build_conv2D_block(global_feature_extractor,filters = 196,kernel_size=3,strides=1)
        conv2d_deconv4   = build_conv2Dtranspose_block(conv2d_deconv5_1, filters=196, kernel_size=4, strides=2)

        #Concat4
        Concat_concat4 = concatenate([conv2d_deconv4, global_feature_extractor1] , axis=-1)

        conv2d_deconv4_1 = build_conv2D_block(Concat_concat4,filters = 128,kernel_size=3,strides=1)
        conv2d_deconv3   = build_conv2Dtranspose_block(conv2d_deconv4_1, filters=128, kernel_size=4, strides=2)
 
        #Concat4
        feature_fusion_main_branch = concatenate([conv2d_deconv3, skip_connection3] , axis=-1)

        feature_fusion_main_branch = build_DepthwiseConv2D_block(feature_fusion_main_branch,filters = 3)

        feature_fusion_main_branch = Conv2D(128, 1, 1, padding="same")(feature_fusion_main_branch)
        feature_fusion_main_branch = BatchNormalization()(feature_fusion_main_branch)

        feature_fusion_skip_connection = Conv2D(128, 1, 1, padding="same")(skip_connection3)
        feature_fusion_skip_connection = BatchNormalization()(feature_fusion_skip_connection)

        feature_fusion = feature_fusion_main_branch + feature_fusion_skip_connection

        # Classifier
        
        classifier = build_SeparableConv2D_block( feature_fusion, 128,3,1)
        classifier = build_SeparableConv2D_block( classifier, 128,3,1)
        classifier   = build_conv2D_block(classifier,filters = 3,kernel_size=1,strides=1)
        #Concat4

        conv2d_deconv3_1 = build_conv2D_block(classifier,filters = 96,kernel_size=3,strides=1)
        conv2d_deconv2   = build_conv2Dtranspose_block(conv2d_deconv3_1, filters=96, kernel_size=4, strides=2)
        conv2d_deconv2 = concatenate([conv2d_deconv2, skip_connection2] , axis=-1)

        conv2d_deconv2_1 = build_conv2D_block(conv2d_deconv2,filters = 64,kernel_size=3,strides=1)
        conv2d_deconv1   = build_conv2Dtranspose_block(conv2d_deconv2_1, filters=64, kernel_size=4, strides=2)
        #Concat4
        conv2d_deconv3 = concatenate([conv2d_deconv1, skip_connection1] , axis=-1)

        conv2d_deconv1_1 = build_conv2D_block(conv2d_deconv3,filters = 24,kernel_size=3,strides=1)
        conv2d_deconv0   = build_conv2Dtranspose_block(conv2d_deconv1_1, filters=24, kernel_size=4, strides=2)

        output = Conv2DTranspose(filters=self.num_class, kernel_size=1, strides=1, activation='softmax', padding='same', name='output')(conv2d_deconv0)
            
        self.model = Model(inputs=inputs, outputs=output)
        if print_summary:
            print(self.model.summary())
        # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
