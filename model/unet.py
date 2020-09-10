import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,UpSampling2D
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
from .net_parts import build_conv2D_block, build_conv2Dtranspose_block

class unet:
    def __init__(self,  print_summary=False,image_size=(512, 512, 3),num_class=3):
        self.parameter = [24,48,96,128,256]
        self.num_class = num_class
        self.build(print_summary=print_summary,image_size=(512, 512, 3),bilinear=False)
        self.batch_generator =  None
        

    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='unet_model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='unet_model.h5'):
        self.model.load_weights(file_path)
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        self.model.fit(self.batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build(self, print_summary=False,image_size=(512, 512, 3), bilinear=True):
        factor = 2 if bilinear else 1
        inputs = Input(shape=image_size)
            
        # initial layer
        conv2d_conv0_1 = build_conv2D_block(inputs,        filters = self.parameter[0],kernel_size=1,strides=1)
        conv2d_conv0   = build_conv2D_block(conv2d_conv0_1,filters = self.parameter[0],kernel_size=3,strides=1)
 
        # first conv layer
        max_d1 = MaxPooling2D()(conv2d_conv0)
        conv2d_conv1_1 = build_conv2D_block(max_d1,  filters = self.parameter[1],kernel_size=3,strides=1)
        conv2d_conv1   = build_conv2D_block(conv2d_conv1_1,filters = self.parameter[1],kernel_size=3,strides=1)
   
        # second conv layer
        max_d2 = MaxPooling2D()(conv2d_conv1)
        conv2d_conv2_2 = build_conv2D_block(max_d2,  filters = self.parameter[2],kernel_size=3,strides=1)
        conv2d_conv2_1 = build_conv2D_block(conv2d_conv2_2,filters = self.parameter[2],kernel_size=3,strides=1)

        # third conv layer
        max_d3 = MaxPooling2D()(conv2d_conv2_1)
        conv2d_conv3_2 = build_conv2D_block(max_d3,  filters = self.parameter[3],kernel_size=3,strides=1)
        conv2d_conv3_1 = build_conv2D_block(conv2d_conv3_2,filters = self.parameter[3],kernel_size=3,strides=1)

        # fourth conv layer
        max_d4 = MaxPooling2D()(conv2d_conv3_1)
        conv2d_conv4_2 = build_conv2D_block(max_d4,  filters = self.parameter[4]/factor,kernel_size=3,strides=1)
        conv2d_conv4_1 = build_conv2D_block(conv2d_conv4_2,filters = self.parameter[4]/factor,kernel_size=3,strides=1)

        if bilinear:
            up4 = UpSampling2D()(conv2d_conv4_1)
        else:
            up4   = build_conv2Dtranspose_block(conv2d_conv4_1, filters=self.parameter[4]/factor, kernel_size=4,strides=2)
        up_conv4 = build_conv2D_block(up4,  filters = self.parameter[4]/factor,kernel_size=3,strides=1)
        up_conv4 = build_conv2D_block(up_conv4,filters = self.parameter[4]/factor,kernel_size=3,strides=1)

        if bilinear:
            up3 = UpSampling2D()(up_conv4)
        else:
            up3   = build_conv2Dtranspose_block(up_conv4, filters=self.parameter[3]/factor, kernel_size=4,strides=2)
        up_conv3 = build_conv2D_block(up3,  filters = self.parameter[3]/factor,kernel_size=3,strides=1)
        up_conv3 = build_conv2D_block(up_conv3,filters = self.parameter[3]/factor,kernel_size=3,strides=1)

        if bilinear:
            up2 = UpSampling2D()(up_conv3)
        else:
            up2   = build_conv2Dtranspose_block(up_conv3, filters=self.parameter[2]/factor, kernel_size=4,strides=2)
        up_conv2 = build_conv2D_block(up2,  filters = self.parameter[2]/factor,kernel_size=3,strides=1)
        up_conv2 = build_conv2D_block(up_conv2,filters = self.parameter[2]/factor,kernel_size=3,strides=1)

        if bilinear:
            up1 = UpSampling2D()(up_conv2)
        else:
            up1   = build_conv2Dtranspose_block(up_conv2, filters=self.parameter[2]/factor, kernel_size=4,strides=2)
        up_conv1 = build_conv2D_block(up1,  filters = self.parameter[2]/factor,kernel_size=3,strides=1)
        up_conv1 = build_conv2D_block(up_conv1,filters = self.parameter[2]/factor,kernel_size=3,strides=1)

        output = Conv2D(filters=self.num_class, kernel_size=1, strides=1, activation='softmax', padding='same', name='output')(up_conv1)
            
        self.model = Model(inputs=inputs, outputs=output)
        if print_summary:
            print(self.model.summary())
        # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
