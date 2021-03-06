import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,Add
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import numpy as np
from PIL import Image
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
from .net_parts import build_conv2D_block, build_conv2Dtranspose_block

class rtnet:
    def __init__(self,  print_summary=False,image_size=(448, 512, 3),num_class=3):
        self.parameter = [24,48,64,96,128,196]
        self.num_class = num_class
        self.build(print_summary=print_summary,image_size=image_size)
        self.batch_generator =  None
        

    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='rt_model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='rt_model.h5'):
        self.model.load_weights(file_path)
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        self.model.fit(self.batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build(self, print_summary=False,image_size=(448, 512, 3)):
        inputs = Input(shape=image_size)
            
        # initial layer
        conv2d_conv0_1 = build_conv2D_block(inputs,        filters = self.parameter[0],kernel_size=1,strides=1)
        conv2d_conv0   = build_conv2D_block(conv2d_conv0_1,filters = self.parameter[0],kernel_size=3,strides=1)
        ###########
        conv2d_conv0   = build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv0_dilation   = build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv0_dilation2   = build_conv2D_block(conv2d_conv0_dilation,filters = self.parameter[0],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv0 = Add()([conv2d_conv0, conv2d_conv0_dilation,conv2d_conv0_dilation2])
        # conv2d_conv0 = concatenate([conv2d_conv0, conv2d_conv0_dilation,conv2d_conv0_dilation2] , axis=-1)
        conv2d_conv0   = build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # first conv layer
        conv2d_conv1_1 = build_conv2D_block(conv2d_conv0,  filters = self.parameter[1],kernel_size=3,strides=2)
        conv2d_conv1   = build_conv2D_block(conv2d_conv1_1,filters = self.parameter[1],kernel_size=3,strides=1)
        ###########
        conv2d_conv1   = build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv1_dilation   = build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv1_dilation2   = build_conv2D_block(conv2d_conv1_dilation,filters = self.parameter[1],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv1 = Add()([conv2d_conv1, conv2d_conv1_dilation,conv2d_conv1_dilation2])
        # conv2d_conv1 = concatenate([conv2d_conv1, conv2d_conv1_dilation] , axis=-1)
        conv2d_conv1   = build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # second conv layer
        conv2d_conv2_2 = build_conv2D_block(conv2d_conv1,  filters = self.parameter[2],kernel_size=3,strides=2)
        conv2d_conv2_1 = build_conv2D_block(conv2d_conv2_2,filters = self.parameter[2],kernel_size=3,strides=1)
        conv2d_conv2   = build_conv2D_block(conv2d_conv2_1,filters = self.parameter[2],kernel_size=3,strides=1)
        ###########
        conv2d_conv2   = build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv2_dilation   = build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv2_dilation2   = build_conv2D_block(conv2d_conv2_dilation,filters = self.parameter[2],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv2 = Add()([conv2d_conv2, conv2d_conv2_dilation,conv2d_conv2_dilation2])
        # conv2d_conv2 = concatenate([conv2d_conv2, conv2d_conv2_dilation] , axis=-1)
        conv2d_conv2   = build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # third conv layer
        conv2d_conv3_2 = build_conv2D_block(conv2d_conv2,  filters = self.parameter[3],kernel_size=3,strides=2)
        conv2d_conv3_1 = build_conv2D_block(conv2d_conv3_2,filters = self.parameter[3],kernel_size=3,strides=1)
        conv2d_conv3   = build_conv2D_block(conv2d_conv3_1,filters = self.parameter[3],kernel_size=3,strides=1)
        ###########
        conv2d_conv3   = build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv3_dilation   = build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv3_dilation2   = build_conv2D_block(conv2d_conv3_dilation,filters = self.parameter[3],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv3 = Add()([conv2d_conv3, conv2d_conv3_dilation,conv2d_conv3_dilation2])
        # conv2d_conv3 = concatenate([conv2d_conv3, conv2d_conv3_dilation] , axis=-1)
        conv2d_conv3   = build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # fourth conv layer
        conv2d_conv4_2 = build_conv2D_block(conv2d_conv3,  filters = self.parameter[4],kernel_size=3,strides=2)
        conv2d_conv4_1 = build_conv2D_block(conv2d_conv4_2,filters = self.parameter[4],kernel_size=3,strides=1)
        conv2d_conv4   = build_conv2D_block(conv2d_conv4_1,filters = self.parameter[4],kernel_size=3,strides=1)
        ###########
        conv2d_conv4   = build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv4_dilation   = build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv4_dilation2   = build_conv2D_block(conv2d_conv4_dilation,filters = self.parameter[4],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv4 = Add()([conv2d_conv4, conv2d_conv4_dilation,conv2d_conv4_dilation2])
        # conv2d_conv4 = concatenate([conv2d_conv4, conv2d_conv4_dilation] , axis=-1)
        conv2d_conv4   = build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # fifth conv layer
        conv2d_conv5_1 = build_conv2D_block(conv2d_conv4,  filters = self.parameter[5],kernel_size=3,strides=2)
        conv2d_conv5   = build_conv2D_block(conv2d_conv5_1,filters = self.parameter[5],kernel_size=3,strides=1)
        ###########
        conv2d_conv5   = build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1,dilation_rate=(3, 3))
        conv2d_conv5_dilation   = build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1,dilation_rate=(5, 5))
        conv2d_conv5_dilation2   = build_conv2D_block(conv2d_conv5_dilation,filters = self.parameter[5],kernel_size=3,strides=1,dilation_rate=(9, 9))
        conv2d_conv5 = Add()([conv2d_conv5, conv2d_conv5_dilation,conv2d_conv5_dilation2])
        # conv2d_conv5 = concatenate([conv2d_conv5, conv2d_conv5_dilation] , axis=-1)
        conv2d_conv5   = build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1,dilation_rate=(3, 3))
        # fifth deconv layer
        conv2d_deconv5_1 = build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1)
        conv2d_deconv4   = build_conv2Dtranspose_block(conv2d_deconv5_1, filters=self.parameter[4], kernel_size=4,strides=2)
            
        #Concat4
        Concat_concat4 = concatenate([conv2d_conv4, conv2d_deconv4] , axis=-1)
            
        # fourth deconv layer
        conv2d_deconv4_1 = build_conv2D_block(Concat_concat4,filters = self.parameter[4],kernel_size=3,strides=1)
        conv2d_deconv3   = build_conv2Dtranspose_block(conv2d_deconv4_1, filters=self.parameter[3], kernel_size=4, strides=2)
            
        #Concat3
        Concat_concat3 = concatenate([conv2d_conv3 , conv2d_deconv3] , axis=-1)
            
        # third deconv layer
        conv2d_deconv3_1 = build_conv2D_block(Concat_concat3,filters = self.parameter[3],kernel_size=3,strides=1)
        conv2d_deconv2   = build_conv2Dtranspose_block(conv2d_deconv3_1, filters=self.parameter[2], kernel_size=4, strides=2)
            
        #Concat2
        Concat_concat2 = concatenate([conv2d_conv2 , conv2d_deconv2] , axis=-1)
            
        # sencod deconv layer
        conv2d_deconv2_1 = build_conv2D_block(Concat_concat2,filters = self.parameter[2],kernel_size=3,strides=1)
        conv2d_deconv1   = build_conv2Dtranspose_block(conv2d_deconv2_1, filters=self.parameter[1], kernel_size=4, strides=2)
            
        #Concat1
        Concat_concat1 = concatenate([conv2d_conv1 , conv2d_deconv1] , axis=-1)
            
        # first deconv layer
        conv2d_deconv1_1 = build_conv2D_block(Concat_concat1,filters = self.parameter[1],kernel_size=3,strides=1)
        conv2d_deconv0   = build_conv2Dtranspose_block(conv2d_deconv1_1, filters=self.parameter[0], kernel_size=4, strides=2)


        output = Conv2DTranspose(filters=self.num_class, kernel_size=1, strides=1, activation='softmax', padding='same', name='output')(conv2d_deconv0)
            
        self.model = Model(inputs=inputs, outputs=output)
        if print_summary:
            print(self.model.summary())
        # self.parallel_model = multi_gpu_model(self.model, gpus=2)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
