#!/bin/python3

# ============================================================================
# Jimut Bahan Pal 
# May, 27, 2021
# A script to generate Mask RCNN Results
# ============================================================================

import os
import cv2
import json
import glob
import math
import random
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



vibrant_colors = [[0,0,255], [0,255,0], [0,255,255], [255,0,0], [255,0,255], [255,255,0]]

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
    return anno


all_folders_root = 'Blood SmearAnalysis'
all_folders = glob.glob('{}/*'.format(all_folders_root))

for folders in tqdm(all_folders):
    all_files_per_folder = glob.glob('{}/*'.format(folders))
    all_valid_files_per_folder = []
    for files in all_files_per_folder:
        get_extension = files.split('.')[-1]
        if get_extension == 'jpg' or get_extension == 'png':
            all_valid_files_per_folder.append(files)
    
    file_name = folders+"/annotations.xml"
    for valid_image_names in all_valid_files_per_folder:
        try:

            valid_image_names_ = valid_image_names.split('/')[-1]
            print("Image Name = ",valid_image_names_)
            annot = parse_anno_file(file_name,valid_image_names_)
            # print("valid image names_ = ",valid_image_names_)
            print("Annotation = ",annot)
            # print("--"*20)
            annot = annot[0]
            # print(json.dumps(annot, indent=4, sort_keys=True))
            im_height = annot['height']
            im_width = annot['width']
            im_id = annot['id']
            im_name = annot['name']
            im_shapes = annot['shapes']
            get_im_path = folders+"/"+im_name
            get_image = cv2.imread(get_im_path,cv2.IMREAD_COLOR)
            # plt.imshow(get_image[:,:,::-1])
            # plt.show()
            # print(im_height)
            # print(im_width)
            # print(im_id)
            # print("Annotation name = ",im_name)
            
            name_ = im_name.split('.')[0]
            # read image as RGB and add alpha (transparency)
            im = Image.open(valid_image_names).convert("RGBA")
            imArray = np.asarray(im)
            count = 0
            get_all_masks = []
            get_bbox_coords = [] # x, y, w, h, label, col
            for shape in im_shapes:
                count += 1
                # save_name = SAVE_FOLDER_NAME+"/"+rev_folder_map[subfolder_name]+"/"+name_+"_"+str(count)+".jpg"
                # print("Save Name = ",save_name)
                # print(shape)
                points = shape['points']  
                label = shape['label']
                print(points)
                all_points = points.split(';')
                # print(all_points)
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
                # plt.imshow(maskIm)
                # plt.show()

                # =========================================
                # create the mask image with certain colour
                
                maskIm = np.array(maskIm) * 255
                # plt.imshow(maskIm)
                # plt.show()
                print(maskIm.max())
                act_mask = np.zeros_like(get_image)
                act_mask[:,:,0] = maskIm
                act_mask[:,:,1] = maskIm
                act_mask[:,:,2] = maskIm
                print("--"*40,get_image.shape,", ",act_mask.shape)
                # mask_image_out = get_image.copy()
                green_mask = get_image.copy()
                col = vibrant_colors[random.randint(0,5)]

                green_mask[(act_mask==255).all(-1)] = col
                get_bbox_coords.append([min_x,min_y,gap_x,gap_y, label, col])
                # plt.imshow(green_mask[:,:,::-1])
                # plt.show()

                get_all_masks.append(green_mask)


                # =========================================
            print("--"*40,len(get_all_masks))
            final_masked_im = np.zeros_like(get_image)
            final_masked_im = float(1/(len(get_all_masks)+1))*get_image
            print("fff",final_masked_im.max())

            for image in get_all_masks:
                print("max = ",image.max(),"min = ",image.min())
                final_masked_im = final_masked_im + float(1/(len(get_all_masks)+1))*image
                print("Final max = ",final_masked_im.max(),"min = ",final_masked_im.min())
                # plt.imshow(final_masked_im[:,:,::-1])
                # plt.show()
            print("fin = ",final_masked_im.max())

            for items in get_bbox_coords:
                x,y,w,h,name, col = int(items[0]), int(items[1]), int(items[2]), int(items[3]), items[4], items[5]
                cropped_img = np.zeros((w,h,3))
                cropped_img = get_image[y:y+h,x:x+w]
                plt.imshow(cropped_img[:,:,::-1])
                plt.show()
                print(x,y,w,h,name)
                #col = vibrant_colors[random.randint(0,5)]
                cv2.rectangle(final_masked_im, (x,y), (x+w,y+h), col,15)
                # img, text, coord, type of font, size, col, thickness
                cv2.putText(final_masked_im, str(name), (x, y), 0, 3, [0,0,0], 10)
            plt.imshow(final_masked_im[:,:,::-1]/255)
            plt.show()
        # cv2.imshow("mask", final_masked_im)
        # cv2.waitKey(0)
        except:
            print("PASS")



