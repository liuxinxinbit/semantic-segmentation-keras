from model.rtnet import rtnet
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import imgviz
import time
from utils.data_process import random_crop_or_pad
from model import rtnet
from utils.dataset import marine_data

def train(md):
    RTNet = rtnet()
    RTNet.batch_generator = md.BatchGenerator(batch_size=4, image_size=(448, 512, 3), labels=3)
    RTNet.train(epochs=10, steps_per_epoch=500)
    RTNet.save()
def test():
    RTNet = rtnet()
    RTNet.load()
    for flag in range(500):
        print(str(flag).zfill(5))
        image = np.float32(Image.open("marine_data/11/images/"+str(flag+1).zfill(5)+".jpg"))/255
        image = random_crop_or_pad(image)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        prediction = RTNet.predict(image)
        result = np.argmax(prediction[0:,:,:],-1)
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.pause(0.01)
        plt.clf()
if __name__ == '__main__':
    md = marine_data()
    train(md)