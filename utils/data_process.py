import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
import numpy as np
import random
import cv2
from PIL import Image

def json2data(json_file):
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
    print(label_names)
    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )
    return img,lbl,lbl_viz
def find_target_file(find_dir,format_name):
    files= [find_dir+file for file in os.listdir(find_dir) if file.endswith(format_name)]
    return files
def read_traindata_names():
    trainset=[]
    for i in range(12):
        find_dir = 'marine_data/'+ str(i+1) + '/images/'
        files = find_target_file(find_dir,'.json')
        trainset+=files
    return trainset
def random_crop_or_pad( image, truth=None, size=(448, 512)):
    if truth is not None:
        assert image.shape[:2] == truth.shape[:2]

    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0],:,:]
        if truth is not None:
            truth = truth[crop_random_y:crop_random_y + size[0],:]
    else:
        zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)
        zeros[:image.shape[0], :image.shape[1], :] = image                                          
        image = np.copy(zeros)
        if truth is not None:
            zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros)

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:,crop_random_x:crop_random_x + size[1],:]
        if truth is not None:
            truth = truth[:,crop_random_x:crop_random_x + size[1]]
    else:
        zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)
        if truth is not None:
            zeros = np.zeros((truth.shape[0], size[1]))
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros)            
    if truth is not None:
        return image, truth
    else:
        return image, truth

def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
def data_augment(xb):
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb

def preprocess(pil_img, pil_label=None,image_size=(512, 512)):
    if pil_label is not None:
        assert pil_img.size == pil_label.size,  'size erros'
    w, h = pil_img.size
    newW, newH = min(w,h), min(w,h)
    assert newW > 0 and newH > 0, 'Scale is too small'
        
    new_label= np.zeros((newW, newH))
    new_img = np.zeros((newW, newH,3))
    pil_img = np.array(pil_img)
    if pil_label is not None:
        pil_label = np.array(pil_label)
    if w>h:
        new_img= pil_img[np.int16((w-h)/2):np.int16((w-h)/2)+h,:]
        if pil_label is not None:
            new_label = pil_label[np.int16((w-h)/2):np.int16((w-h)/2)+h,:]
    elif h>w:
        new_img = pil_img[:,np.int16((h-w)/2):np.int16((h-w)/2)+w]
        if pil_label is not None:
            new_label= pil_label[:,np.int16((h-w)/2):np.int16((h-w)/2)+w]
    else:
        new_img=pil_img
        if pil_label is not None:
            new_label=pil_label
    new_label = Image.fromarray(new_label.astype('uint8'))
    new_img = Image.fromarray(new_img.astype('uint8')).convert('RGB')
    new_img = new_img.resize(image_size)
    new_label = new_label.resize(image_size)
    img_nd = np.array(new_img)
    label_nd = np.array(new_label)

    # if len(img_nd.shape) == 2:
    #     img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    # img_trans = img_nd.transpose((2, 0, 1))
    # if img_nd.max() > 1:
    #     img_trans = img_nd / 255
    img_nd = data_augment(img_nd)
    return img_nd,label_nd



# files_list=[]
# for i in range(12):
#     find_dir = 'marine_data/'+ str(i+1) + '/images/'
#     files = find_target_file(find_dir,'.json')
#     files_list+=files


# for json_file in files_list:
#     img,lbl,lbl_viz = json2data(json_file)
#     print(img.shape)
#     plt.imshow(lbl_viz)

#     plt.show()
# batch_size=8
# image_size=(448, 512, 3)
# labels=3
# trainset  = read_traindata_names()
# while True:
#     images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
#     truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
#     for i in range(batch_size):
#         random_line = random.choice(trainset)
#         image,truth_mask,lbl_viz = json2data(random_line)
#         truth_mask=truth_mask
#         image, truth = random_crop_or_pad(image, truth_mask, image_size)
#         images[i] = image/255
#         truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
#         print(image.shape,truth.shape)
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.subplot(1, 2, 2)
#         plt.imshow(truth)
#         plt.pause(1)
#         plt.clf()