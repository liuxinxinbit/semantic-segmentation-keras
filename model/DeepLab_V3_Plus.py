import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,\
    LeakyReLU,AveragePooling2D,DepthwiseConv2D,ZeroPadding2D, Add,UpSampling2D,Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import multi_gpu_model
import random
import numpy as np
from PIL import Image
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
from math import ceil
from .net_parts import build_conv2D_block, build_conv2Dtranspose_block,bottleneck,pyramid_pooling,build_SeparableConv2D_block,build_DepthwiseConv2D_block,ASPP,Decoder
from backbone import ResNet
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

class DeepLab_V3_Plus:
    def __init__(self,  print_summary=False,image_size=(512, 512, 3),num_class=3):
        self.num_class = num_class
        self.build(print_summary=print_summary,image_size=image_size)
        self.batch_generator =  None
        
    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='deeplabv3plus_model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='deeplabv3plus_model.h5'):
        self.model.load_weights(file_path)
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        self.model.fit(self.batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build(self, backbone='resnet', output_stride=16, print_summary=False,image_size=(512, 512, 3), num_classes=3,resnet_layers=50):
        inputdata = Input(shape=image_size)
        x, low_level_feat = ResNet(inputdata=inputdata, layers=[
                                   3, 4, 23, 3], output_stride=output_stride)
        x = ASPP(x, output_stride=16)
        x = Decoder(x,low_level_feat,self.num_class)     

        x = Interp(x,[image_size[0],image_size[1]])
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
# DeepLab_V3_Plus = DeepLab_V3_Plus(image_size = (512, 512, 3),num_class=3,print_summary=True)
