from model.DeepLab_V3_Plus import DeepLab_V3_Plus
from utils.eval import SegmentationMetric
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import imgviz
import time
from utils.data_process import preprocess, random_crop_or_pad
from model import rtnet
from utils.dataset import marine_data,voc_data

image_size=(512, 512, 3)
num_class=3
def train(md):
    DeeplabNet = DeepLab_V3_Plus(image_size = image_size,num_class=num_class)
    DeeplabNet.batch_generator = md.BatchGenerator(batch_size=2, image_size=(512, 512, 3), labels=num_class)
    DeeplabNet.train(epochs=10, steps_per_epoch=500)
    DeeplabNet.save()
def test():
    DeeplabNet = DeepLab_V3_Plus(image_size = image_size,num_class=num_class)
    DeeplabNet.load()
    for flag in range(500):
        print(str(flag).zfill(5))
        image = Image.open("../marine_data/11/images/"+str(flag+1).zfill(5)+".jpg")
        image,label = preprocess(image)
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image))
        prediction = DeeplabNet.predict(image/255)
        result = np.argmax(prediction[0,:,:,:],-1)
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.pause(0.01)
        plt.clf()

def _eval(dataset):
    images, truths = dataset.eval_data(batch_size=8,image_size=(512, 512, 3),labels=num_class)
    print(truths.shape)
    DeeplabNet = DeepLab_V3_Plus(image_size = image_size,num_class=num_class)
    DeeplabNet.load()
    mean_acc=0
    mean_mIoU = 0
    for i in range(images.shape[0]):
        prediction = DeeplabNet.predict(images[i,:,:,:])
        metric = SegmentationMetric(num_class)
        prediction = np.argmax(prediction[0,:,:,:],-1)
        # plt.imshow(prediction)
        # plt.show()
        truth = np.argmax(truths[i,:,:,:],-1)
        metric.addBatch(prediction, truth)
        acc = metric.pixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        mean_acc+=acc
        mean_mIoU+=mIoU
    print(mean_acc/images.shape[0], mean_mIoU/images.shape[0])

if __name__ == '__main__':
    #marine data******************
    num_class = 3
    md = marine_data()
    # train(md)
    test()
    _eval(md)
    #voc data*********************
    # num_class = 21
    # voc = voc_data()
    # train(voc)
    # _eval(voc)
    #*****************************
