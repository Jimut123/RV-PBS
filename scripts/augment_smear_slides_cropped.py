"""
Jimut Bahan Pal
31-May-2021
A Script to Augment the Smear Slides Cropped Dataset.
"""

import os
import cv2
import json
import glob
import math
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from lxml import etree
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from skimage.util import random_noise



print("LOADING BG TEXTURES...")
NORMAL_BG_NAMES = glob.glob('normal_bg/*.jpg')
STAIN_BG_NAMES = glob.glob('stain_bg/*.jpg')


def _create_LUT_8UC1(x, y):
  spl = UnivariateSpline(x, y)
  return spl(range(256))

def get_random_vector():
    # to generate a random one hot vector of size
    # positions are given by vectors
    # 0 - apply_normal_background_image(img) or apply_stain_background_image(img)
    # 1 - zoom_out_image(img, zoom_percentage)
    # 2 - flip_image(img, opt)
    # 3 - apply_color_transformation_image(img, val1, val2, val3)
    # 4 - gaussian_blur_repeatitive_image(img, kernel_size,n_iter)
    # 5 - rotate_image(img, angle)
    # 6 - change_contrast_image(img, alpha)
    # 7 - change_brightness_image(img, beta)
    # 8 - apply_noise_gaussian_image(img)
    # 9 - apply_noise_salt_and_pepper(img, amount)
    random_vector = []
    for i in range(11):
        random_vector.append(random.randint(0,1))
    print(random_vector)
    print(len(random_vector))
    return random_vector


# some of the functions to do image augmentation for augmenting the entire 
# dataset for performing domain adaptation


# This functions should be added before any step
def apply_normal_background_image(img):
    # print(img[0][0][0])
    background_image = cv2.imread(NORMAL_BG_NAMES[random.randint(0,(len(NORMAL_BG_NAMES)-1))])
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    im_h, im_w, n_c = img.shape
    background_image = cv2.resize(background_image,(im_w,im_h))
    # print(background_image.shape)
    # print(img.shape)
    # median blur to remove noises
    img_blurred = cv2.medianBlur(img, 111)
    background_image[img_blurred > 0] = 0
    background_added_image = background_image + img
    return background_added_image

# This functions should be added before any step
def apply_stain_background_image(img):
    # apply background with possible stain
    background_image = cv2.imread(STAIN_BG_NAMES[random.randint(0,(len(STAIN_BG_NAMES)-1))])
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    im_h, im_w, n_c = img.shape
    background_image = cv2.resize(background_image,(im_w,im_h))
    # print(background_image.shape)
    # print(img.shape)
    # median blur to remove noises
    img_blurred = cv2.medianBlur(img, 111)
    background_image[img_blurred > 0] = 0
    background_added_image = background_image + img
    return background_added_image


def zoom_out_image(img, zoom_percentage):
    # zooms out an image by certain percentage
    # please keep this between 0.5 - 0.9
    im_h, im_w, im_c = img.shape
    resized_height, resized_width = int(zoom_percentage*im_h), int(zoom_percentage*im_w)
    # print(resized_height,",",resized_width)
    img = cv2.resize(img, (resized_height,resized_width))
    # make the image to original size
    delta_w = im_h - resized_height
    delta_h = im_w - resized_width
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im



def flip_image(img, opt):
    # filp an image = horizontally or vertically or both
    # 0 for horizontal flip, 1 for vertical flip, -1 for both
    return cv2.flip(img, opt)


def apply_color_transformation_image(img, val1, val2, val3):
    # apply colour transformation to image, generally bluish by certain variant
    # or transform an image to certain colour, specially with a tinge of yellow, and blue
    # this can also be considered as a warm filter, and a cold filter changed according to 
    # value
    decr_ch_lut = _create_LUT_8UC1(val1, val2)
    incr_ch_lut = _create_LUT_8UC1(val1, val3)
    c_r, c_g, c_b = cv2.split(img)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img_rgb = cv2.merge((c_r, c_g, c_b))
    # decrease color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
    

def gaussian_blur_repeatitive_image(img, kernel_size,n_iter):
    # apply gaussian blur with different kernel size and
    # repeat it different times in an image
    for iter in range(n_iter):
        img = cv2.GaussianBlur(img, kernel_size, cv2.BORDER_DEFAULT)    
    return img


def rotate_image(img, angle):
    # rotate an image by certain angle 
    row,col,channel = img.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(img, rot_mat, (col,row))
    return new_image


# def zoom_in_image(img, zoom_percentage):
#     # zoom in an image by certain percentage, doesn't make 
#     # any sense here...
#     pass


def change_contrast_image(img, alpha):
    # change contrast of an image
    # alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted
    

def change_brightness_image(img, beta):
    # Change brightness of an image
    alpha = 1.5 # Contrast control (1.0-3.0)
    # beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def apply_noise_gaussian_image(img):
    # apply salt and pepper noise in an image
    noise_img = random_noise(img, mode='gaussian', seed=None, clip=True)

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    # print(noise_img.max(), noise_img.min())
    return noise_img


def apply_noise_salt_and_pepper(img, amount):
    # apply salt and pepper noise to the image

    noise_img = random_noise(img, mode='s&p',amount=0.3)

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img



def show_img(img):
    plt.imshow(img)
    plt.show()


folders = glob.glob('classification_data/*')
print(folders)

for folder in folders:
    image_files = glob.glob('{}/*'.format(folder))
    for image_name in image_files:
        if 'yml' not in image_name:
            image = cv2.imread(image_name,cv2.IMREAD_COLOR)

            for iter_ in tqdm(range(100)):
                vector_ = get_random_vector()

                dummy_count = 0
                for choice in vector_:
                    # make the choices according to the vector
                    if choice == 0 and dummy_count == 0:
                        # apply normal background to the image
                        image_out = apply_normal_background_image(image)
                    if choice == 1 and dummy_count == 0:
                        # apply stain background to the image
                        image_out = apply_stain_background_image(image)
                    
                    if choice == 1 and dummy_count == 1:
                        # apply zoom out on an image
                        # generates random value between 0.49 to 1.0
                        # probably will be better if this is aliased with other function
                        gen_random = (random.randint(7,10)/10)*(random.randint(7,10)/10)
                        image_out = zoom_out_image(image_out, gen_random)
                    
                    if choice == 1 and dummy_count == 2:
                        image_out =  flip_image(image_out, opt=random.randint(-1,1))
                    
                    if choice == 1 and dummy_count == 3:
                        get_ch = random.randint(0,5)
                        # applying cooler transformations
                        if get_ch == 0:
                            image_out =  apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 30,  80, 120, 192], [0, 40, 95, 142.5, 208])
                        if get_ch == 1:
                            image_out = apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 30,  80, 120, 192], [0, 70, 140, 210, 256])
                        if get_ch == 2:
                            image_out = apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 70,  100, 140, 192], [0, 110, 180, 210, 256])
                        
                        # apply warmer transofrmations
                        if get_ch == 3:
                            image_out = apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 70,  100, 140, 192], [0, 110, 180, 210, 256])
                        if get_ch == 4:
                            image_out = apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 70, 140, 210, 256], [0, 30,  80, 120, 192])
                        if get_ch == 5:
                            image_out = apply_color_transformation_image(image_out, [0, 64, 128, 192, 256], [0, 110, 180, 210, 256], [0, 70,  100, 140, 192])
                    
                    if choice == 1 and dummy_count == 4:
                        # apply gaussian blur on images till limit
                        get_ch = random.randint(0,2)
                        if get_ch == 0:
                            image_out = gaussian_blur_repeatitive_image(image_out, (5,5), 15)
                        if get_ch == 1:
                            image_out = gaussian_blur_repeatitive_image(image_out, (11,11), 15)
                        if get_ch == 2:
                            image_out = gaussian_blur_repeatitive_image(image_out, (17,17), 15)
                    
                    if choice == 1 and dummy_count == 5:
                        # rotate an image by certain angle

                        get_ch = random.randint(0,3)
                        if get_ch == 0:
                            image_out = rotate_image(image_out, random.randint(0,90))
                        if get_ch == 1:
                            image_out = rotate_image(image_out, random.randint(90,180))
                        if get_ch == 2:
                            image_out = rotate_image(image_out, random.randint(180,270))
                        if get_ch == 3:
                            image_out = rotate_image(image_out, random.randint(270,360))
                    
                    if choice == 1 and dummy_count == 6:
                        # change the contrast of the image
                        # change alpha from (1.0-3.0) by a random number
                        get_random = (random.randint(100,173)/100) * (random.randint(100,173)/100)
                        image_out = change_contrast_image(image_out, get_random)
                    
                    if choice == 1 and dummy_count == 7:
                        # Change the brightness of the image
                        image_out = change_brightness_image(image_out, random.randint(1,75))
                        pass

                    if choice == 1 and dummy_count == 8:
                        # apply Gaussian Noise
                        image_out = apply_noise_gaussian_image(image_out)
                    
                    if choice == 1 and dummy_count == 9:
                        # apply salt and pepper noise
                        amount = 1
                        image_out = apply_noise_salt_and_pepper(image_out, amount)
                    dummy_count += 1
                
                # show_img(image_out)
                save_name = image_name.split('.')[0]+"_"+str(iter_)+".jpg"
                cv2.imwrite(save_name,image_out)




