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
from lxml import etree
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def _create_LUT_8UC1(x, y):
  spl = UnivariateSpline(x, y)
  return spl(range(256))


# some of the functions to do image augmentation for augmenting the entire 
# dataset for performing domain adaptation


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


def apply_noise_gaussian_image(img, mean, var, sigma):
    # apply salt and pepper noise in an image
    row,col,ch= img.shape
    gauss = np.random.normal(mean,sigma,(row,col,ch))*255
    # print(gauss.max())
    # print(img.max())
    gauss = gauss.reshape(row,col,ch)
    noisy = 0.15*img + 0.85*gauss
    noisy_img_clipped = np.clip(noisy, 0, 255)  # we might get out of bounds due to noise
    print(noisy_img_clipped.max(),noisy_img_clipped.min())
    return noisy_img_clipped


def apply_noise_salt_and_pepper(img, s_vs_p, amount):
    # apply a background by certain amount on the image
    # the background generally contains RBCs
    row,col,ch = img.shape
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[np.array(coords)] = 255
    # Pepper mode
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[np.array(coords)] = 0
    return out


def apply_normal_background_image(img):
    pass


def apply_stain_background_image(img):
    # apply background with possible stain
    pass


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

            # show_img(flip_image(image, opt=0))
            # show_img(flip_image(image, opt=1))
            # show_img(flip_image(image, opt=-1)) 
            # [0, 30,  80, 120, 192]

            # # applying cooler transformations
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 30,  80, 120, 192], [0, 40, 95, 142.5, 208]))
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 30,  80, 120, 192], [0, 70, 140, 210, 256]))
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 70,  100, 140, 192], [0, 110, 180, 210, 256]))
            
            # applying cooling transformations
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 40, 95, 142.5, 208], [0, 30,  80, 120, 192]))
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 70, 140, 210, 256], [0, 30,  80, 120, 192]))
            # show_img(apply_color_transformation_image(image, [0, 64, 128, 192, 256], [0, 110, 180, 210, 256], [0, 70,  100, 140, 192]))
            
            # show_img(gaussian_blur_repeatitive_image(image, (5,5), 15))
            # show_img(gaussian_blur_repeatitive_image(image, (11,11), 15))
            # show_img(gaussian_blur_repeatitive_image(image, (17,17), 15))

            # show_img(rotate_image(image, random.randint(0,90)))
            # show_img(rotate_image(image, random.randint(90,180)))
            # show_img(rotate_image(image, random.randint(180,270)))
            # show_img(rotate_image(image, random.randint(270,360)))
            
            # for zoom_out in range(10):
            #     # generates random value between 0.49 to 1.0
            #     # probably will be better if this is aliased with other function
            #     gen_random = (random.randint(7,10)/10)*(random.randint(7,10)/10)
            #     show_img(zoom_out_image(image, gen_random)

            # for contrast in range(100):
            #     # change alpha from (1.0-3.0) by a random number
            #     get_random = (random.randint(100,173)/100) * (random.randint(100,173)/100)
            #     # print(get_random)
            #     #show_img(change_contrast_image(image, get_random))
            
            # for brightness in range(10):
            #     show_img(change_brightness_image(image, random.randint(1,75)))
            # mean = 0
            # var = 0.01
            # sigma = var**0.5
            # show_img(apply_noise_gaussian_image(image, mean, var, sigma))

            s_vs_p = 0.5
            amount = 0.004
            show_img(apply_noise_salt_and_pepper(image, s_vs_p, amount))
            break
    break



