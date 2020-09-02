import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
import os
import random
import time
import numpy as np
import cv2
from PIL import Image
import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
from data_process import random_crop_or_pad

class marine_data:
    def __init__(self,data_dir='../marine_data/'):
        self.trainset  = self.read_traindata_names(data_dir)
        self.num_train = len(self.trainset)
        

    def read_traindata_names(self,data_dir):
        trainset=[]
        for i in range(12):
            find_dir = data_dir + str(i+1) + '/images/'
            files = self.find_target_file(find_dir,'.json')
            trainset+=files
        return trainset

    def json2data(self, json_file):
        data = json.load(open(json_file))
        imageData = data.get('imageData')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
               label_value = label_name_to_value[label_name]
            else:
               label_value = len(label_name_to_value)
               label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
           img.shape, data['shapes'], label_name_to_value
        )
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
        )
        return img,lbl,lbl_viz
    def find_target_file(self,find_dir,format_name):
        files= [find_dir+file for file in os.listdir(find_dir) if file.endswith(format_name)]
        return files


    def BatchGenerator(self,batch_size=8, image_size=(448, 512, 3), labels=3):#500, 375
        while True:
            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
            for i in range(batch_size):
                random_line = random.choice(self.trainset)
                image,truth_mask,lbl_viz = self.json2data(random_line)
                truth_mask=truth_mask+1
                image, truth = random_crop_or_pad(image, truth_mask, image_size)
                images[i] = image/255
                truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
            yield images, truths



class voc_data:
    def __init__(self,data_dir='../marine_data/'):
        self.filename = "../dataset/VOCdevkit/VOC2007/"
        self.trainset  = self.read_traindata_names(data_dir)
        self.num_train = len(self.trainset)

    def voc_fileset(self,data_type="train"):
        if data_type==data_type
        lines_img = []
        lines_label = []
        with open(self.filename+"ImageSets/Segmentation/test.txt", 'r') as file_to_read:
            while True:
                line = file_to_read.readline() # 整行读取数据
                if not line:
                    break
                lines_img.append(self.filename+"JPEGImages/"+line[:-1]+".jpg")
                lines_label.append(self.filename+"SegmentationClass/"+line[:-1]+".png")
        return lines_img,lines_label

    def BatchGenerator(self,batch_size=8, image_size=(448, 512, 3), labels=3):#500, 375
        while True:
            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
            for i in range(batch_size):
                random_line = random.choice(self.trainset)
                image,truth_mask,lbl_viz = self.json2data(random_line)
                truth_mask=truth_mask+1
                image, truth = random_crop_or_pad(image, truth_mask, image_size)
                images[i] = image/255
                truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
            yield images, truths

