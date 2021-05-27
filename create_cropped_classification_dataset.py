#!/bin/python3

# ============================================================================
# Jimut Bahan Pal 
# May, 11, 2021
# A script to collect all the slides and convert to a classification dataset
# by parsing each of the annotation files from each of the folders.
# ============================================================================

import os
import cv2
import json
import glob
import math
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Make the classification dataset's directory

SAVE_FOLDER_NAME = 'classification_data'
if not os.path.exists(SAVE_FOLDER_NAME):
    os.makedirs(SAVE_FOLDER_NAME)

def rgba2rgb( rgba, background=(0,0,0) ):
    """
    Converts a given rgba image to rgb
    """
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8')

def parse_anno_file(cvat_xml,image_name):
    """
    Parses annotation file and returns the details of annotation 
    for the given image ID
    """
    root = etree.parse(cvat_xml).getroot()
    # print(root)
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
        # print("Image tag = ",image_tag)
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            # print("box = ",box)
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    # print("Annotation:",anno)
    return anno


folder_map = {}
rev_folder_map = {}
folder_map = {
        "band": "BAND CELLS",
        "basophil": "BASOPHILS",
        "blast": "BLAST CELLS",
        "eosinophil": "EOSINOPHILS",
        "lymphocyte": "LYMPHOCYTES",
        "metamyelocyte": "METAMYELOCYTES",
        "monocyte": "MONOCYTES",
        "myelocyte": "MYELOCYTE",
        "neutrophil": "NEUTROPHILS",
        "promyelocyte": "PROMYELOCYTES"
        }

#print(folder_map)

for item in folder_map:
    rev_folder_map[folder_map[item]] = item

#print(rev_folder_map)



all_folders_root = 'Blood SmearAnalysis'
all_folders = glob.glob('{}/*'.format(all_folders_root))

for folders in tqdm(all_folders):
    # print(folders)
    all_files_per_folder = glob.glob('{}/*'.format(folders))
    #print(all_files_per_folder)
    all_valid_files_per_folder = []
    # annotation_file = ''
    for files in all_files_per_folder:
        #print(files)
        get_extension = files.split('.')[-1]
        #print(get_extension)
        if get_extension == 'jpg' or get_extension == 'png':
            all_valid_files_per_folder.append(files)
        # elif get_extension == 'xml':
        #     annotation_file = files
            #print("*"*40,get_extension)
    # print(all_files_per_folder)
    # print("**"*20,annotation_file)

    file_name = folders+"/annotations.xml"
    # print("oo"*50,file_name)
    for valid_image_names in all_valid_files_per_folder:
        subfolder_name = valid_image_names.split('/')[1]
        # print("--"*50,subfolder_name)
        subfolder_save = SAVE_FOLDER_NAME+"/"+rev_folder_map[subfolder_name]
        if not os.path.exists(subfolder_save):
            os.makedirs(subfolder_save)
        valid_image_names_ = valid_image_names.split('/')[-1]
        # print("Image Name = ",valid_image_names_)
        annot = parse_anno_file(file_name,valid_image_names_)
        # print("valid image names_ = ",valid_image_names_)
        # print("Annotation = ",annot)
        # print("--"*20)
        annot = annot[0]
        # print(json.dumps(annot, indent=4, sort_keys=True))
        im_height = annot['height']
        im_width = annot['width']
        im_id = annot['id']
        im_name = annot['name']
        im_shapes = annot['shapes']
        # print(im_height)
        # print(im_width)
        # print(im_id)
        # print("Annotation name = ",im_name)
        
        name_ = im_name.split('.')[0]
        # read image as RGB and add alpha (transparency)
        im = Image.open(valid_image_names).convert("RGBA")
        imArray = np.asarray(im)
        count = 0
        for shape in im_shapes:
            count += 1
            save_name = SAVE_FOLDER_NAME+"/"+rev_folder_map[subfolder_name]+"/"+name_+"_"+str(count)+".jpg"
            # print("Save Name = ",save_name)
            #print(shape)  
            points = shape['points']  
            #print(points)
            all_points = points.split(';')
            #print(all_points)
            x_y = []
            all_x = []
            all_y = []
            for point_ in all_points:
                x = float(point_.split(',')[0])
                y = float(point_.split(',')[1])
                all_x.append(x)
                all_y.append(y)
                #print("X = ",x," Y = ",y)
                x_y.append((x,y))
            # print(x_y)
            max_x = max(all_x)
            min_x = min(all_x)
            max_y = max(all_y)
            min_y = min(all_y)
            gap_x = max_x - min_x
            gap_y = max_y - min_y

            # print(max_x, "  ", min_x, "  ",max_y, " ",min_y)
            maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
            ImageDraw.Draw(maskIm).polygon(x_y, outline=1, fill=1)
            mask = np.array(maskIm)

            # assemble new image (uint8: 0-255)
            newImArray = np.empty(imArray.shape,dtype='uint8')

            # colors (three first columns, RGB)
            newImArray[:,:,:3] = imArray[:,:,:3]
            # transparency (4th column)
            newImArray[:,:,3] = mask*255
            # plt.imshow(newImArray)
            # plt.show()
            img_extract = np.zeros((math.ceil(gap_x),math.ceil(gap_y),3))
            img_extract = newImArray[math.ceil(min_y):math.ceil(max_y),math.ceil(min_x):math.ceil(max_x)]
            # plt.imshow(img_extract)
            # plt.show()
            # back to Image from numpy
            newIm = Image.fromarray(newImArray, "RGBA")

            img_extract = rgba2rgb(img_extract)
            # print(img_extract.shape)
            cv2.imwrite(save_name,np.array(img_extract))
            # newIm.save(save_name)
            # break


