'''
Name: trait_extract_parallel.py

Version: 1.0

Summary: Extract plant shoot traits (larea, kernel_area_ratio, max_width, max_height, avg_curv, color_cluster) by paralell processing 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2022-09-29

USAGE:

time python3 trait_computation_mazie_ear.py -p ~/example/plant_test/seeds/test/ -ft png -s Ycc -c 2 -min 250000

'''

# import necessary packages
import os
import glob
import utils

from collections import Counter

import argparse

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from skimage.feature import peak_local_max
from skimage.morphology import medial_axis
from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
from skimage import measure
from skimage.color import rgb2lab, deltaE_cie76
from skimage import morphology
from skimage.segmentation import clear_border, watershed
from skimage.measure import regionprops

from scipy.spatial import distance as dist
from scipy import optimize
from scipy import ndimage
from scipy.interpolate import interp1d

from skan import skeleton_to_csgraph, Skeleton, summarize, draw

import imutils
from imutils import perspective

import numpy as np
import argparse
import cv2

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import collections

import math
import openpyxl
import csv
    
from tabulate import tabulate

from pathlib import Path 
from pylibdmtx.pylibdmtx import decode
import re


import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing



import warnings
warnings.filterwarnings("ignore")



MBFACTOR = float(1<<20)


# generate foloder to store the output results
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False
        

# sort contoures based on method
def sort_contours(cnts, method="left-to-right"):
    
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts


# segment mutiple objects in image, for maize ear image, based on the protocal, shoudl be two objects. 
def mutilple_objects_seg(orig):
    
    shifted = cv2.pyrMeanShiftFiltering(orig, 21, 70)

    height, width, channels = orig.shape

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Taking a matrix of size 25 as the kernel
    kernel = np.ones((25,25), np.uint8)
    
    thresh_dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    thresh_erosion = cv2.erode(thresh, kernel, iterations=1)
    
    #define result path for labeled images
    #result_img_path = save_path_label + str(filename[0:-4]) + '_thresh.jpg'
    #cv2.imwrite(result_img_path, thresh_erosion)
    
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh_erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)[0:2]

    cnts_sorted = sort_contours(cnts_sorted, method="left-to-right")
    
    
    #c = max(cnts, key=cv2.contourArea)
    center_locX = []
    center_locY = []
    cnt_area = [0] * 2
    
    img_thresh = np.zeros(orig.shape, np.uint8)
    
    img_overlay_bk = orig
    
    
    
    for idx, c in enumerate(cnts_sorted):
        
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        center_locX.append(cX)
        center_locY.append(cY)
        
        #cnt_area.append(cv2.contourArea(c))
        
        cnt_area[idx] = cv2.contourArea(c)
        
        # draw the contour and center of the shape on the image
        img_overlay = cv2.drawContours(img_overlay_bk, [c], -1, (0, 255, 0), 2)
        mask_seg = cv2.drawContours(img_thresh, [c], -1, (255,255,255),-1)
        #center_result = cv2.circle(img_thresh, (cX, cY), 14, (0, 0, 255), -1)
        img_overlay = cv2.putText(img_overlay_bk, "{}".format(idx +1), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 5.5, (255, 0, 0), 5)
        
    
    #print(center_locX, center_locY)
    
    divide_X = int(sum(center_locX) / len(center_locX))
    divide_Y = int(sum(center_locY) / len(center_locY))
    
    #print(divide_X, divide_Y)
    
    
    #center_result = cv2.circle(image, (divide_X, divide_Y), 14, (0, 255, 0), -1)
    
    
    #define result path for labeled images
    #result_img_path = save_path_label + str(filename[0:-4]) + '_center.jpg'
    #cv2.imwrite(result_img_path, center_result)
    
    
    
    left_img = orig[0:height, 0:divide_X]

    
    
    right_img = orig[0:height, divide_X:width]

    
    mask_seg_gray = cv2.cvtColor(mask_seg, cv2.COLOR_BGR2GRAY)
    
    #(mask_seg_binary, im_bw) = cv2.threshold(mask_seg_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    return left_img, right_img, mask_seg_gray, img_overlay, cnt_area


# color clustering based object segmentation
def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters):
    
    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    else:
        colorSpace = 'bgr'  # set for file naming purposes
        
        
    #image = cv2.pyrMeanShiftFiltering(image, 21, 70)

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)
            
    (width, height, n_channel) = image.shape
    
    #print("image shape: \n")
    #print(width, height, n_channel)
    
    
    
    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    

    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    
    #define number of cluster
    numClusters = max(2, args_num_clusters)
    
    # clustering method
    kmeans = KMeans(n_clusters = numClusters, n_init = 40, max_iter = 500).fit(reshaped)
    
    # get lables 
    pred_label = kmeans.labels_
    
    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)],key = lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i
    
    ret, thresh = cv2.threshold(kmeansImage,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #thresh_cleaned = (thresh)
    
    
    if np.count_nonzero(thresh) > 0:
        
        thresh_cleaned = clear_border(thresh)
    else:
        thresh_cleaned = thresh
    
     
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh_cleaned, connectivity = 8)

    # stats[0], centroids[0] are for the background label. ignore
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    sizes = stats[1:, cv2.CC_STAT_AREA]
    
    Coord_left = stats[1:, cv2.CC_STAT_LEFT]
    
    Coord_top = stats[1:, cv2.CC_STAT_TOP]
    
    Coord_width = stats[1:, cv2.CC_STAT_WIDTH]
    
    Coord_height = stats[1:, cv2.CC_STAT_HEIGHT]
    
    Coord_centroids = centroids
    
    #print("Coord_centroids {}\n".format(centroids[1][1]))
    
    #print("[width, height] {} {}\n".format(width, height))
    
    nb_components = nb_components - 1
    
    
    
    max_size = width*height*0.1
    
    img_thresh = np.zeros([width, height], dtype=np.uint8)
    
    #for every component in the image, keep it only if it's above min_size
    for i in range(0, nb_components):
        
        if (sizes[i] >= min_size):
        
            img_thresh[output == i + 1] = 255
        
    #from skimage import img_as_ubyte
    
    #img_thresh = img_as_ubyte(img_thresh)
    
    #print("img_thresh.dtype")
    #print(img_thresh.dtype)
    
    
    #if mask contains mutiple non-conected parts, combine them into one. 
    contours, hier = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 1:
        
        print("mask contains mutiple non-conected parts, combine them into one\n")
        
        kernel = np.ones((10,10), np.uint8)

        dilation = cv2.dilate(img_thresh.copy(), kernel, iterations = 1)
        
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        
        img_thresh = closing

    
    #return img_thresh
    return img_thresh
    
    #return thresh_cleaned
    

# compute percentage vale 
def percentage(part, whole):
  
  #percentage = "{:.0%}".format(float(part)/float(whole))
  
  percentage = "{:.2f}".format(float(part)/float(whole))
  
  return str(percentage)



# compute middle points of two points in 2D 
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



def comp_external_contour(orig, thresh, img_overlay):
    
    #find contours and get the external one
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    img_height, img_width, img_channels = orig.shape
   
    index = 1
    
    trait_img = orig.copy()
    
    crop_img = orig.copy()
    
    area = 0
    kernel_area_ratio = 0
    w=h=0
    
    #print("length of contours = {}\n".format(len(contours)))
    ####################################################################################
    
    
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    contours_sorted = sort_contours(contours, method="left-to-right")
    
    #c_max = max(contours, key = cv2.contourArea)
    
    area_c_cmax = 0
    
    area_holes_sum = 0
    
    cnt_area = [0] * 2
    
    cnt_width = []
    cnt_height = []
    
    orig = img_overlay
    ###########################################################################
    for index, c in enumerate(contours_sorted):
    #for c in contours:

        if index < 2:
            
            #get the bounding rect
            (x, y, w, h) = cv2.boundingRect(c)
            
            trait_img = cv2.drawContours(orig, c, -1, (255, 255, 0), 1)
            
            
            #roi = orig[y:y+h, x:x+w]
            
            print("ROI {} detected ...\n".format(index+1))
            
            # draw a green rectangle to visualize the bounding rect
            trait_img = cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 4)
            
            '''
            ####################################################################
            margin = 150

            # define crop region
            start_y = int((y - margin) if (y - margin )> 0 else 0)

            start_x = int((x - margin) if (x - margin )> 0 else 0)

            crop_width = int((x + margin + w) if (x + margin + w) < img_width else (img_width))

            crop_height = int((y + margin + h) if (y + margin + h) < img_height else (img_height))

            crop_img = crop_img[start_y:crop_height, start_x:crop_width]

            ######################################################################
            '''
            
            # compute the center of the contour
            M = cv2.moments(c)
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # draw the center of the shape on the image
            #trait_img = cv2.circle(orig, (cX, cY), 7, (255, 255, 255), -1)
            #trait_img = cv2.putText(orig, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            #################################################################################
            '''
            # compute the (rotated) bounding box of the contour to get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            
            # convert all coordinates floating point values to int
            box = np.int0(box)
            
            #draw a blue 'nghien' rectangle
            trait_img = cv2.drawContours(orig, [box], -1, (255, 0, 0), 3)
            
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            
            # draw the midpoints on the image
            trait_img = cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            
            # draw lines between the midpoints
            trait_img = cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            trait_img = cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
            
            
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # draw the object sizes on the image
            #cv2.putText(orig, "{:.0f} pixels".format(dA), (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            #cv2.putText(orig, "{:.0f} pixels".format(dB), (int(trbrX), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            '''
            ###############################################################################

            # the midpoint between bottom-left and bottom-right coordinates
            #(tl, tr, br, bl) = box
            
            tl = (x, y+h*0.5)
            tr = (x+w, y+h*0.5)
            br = (x+w*0.5, y)
            bl = (x+w*0.5, y+h)
            
            # compute the midpoint between bottom-left and bottom-right coordinates
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            
            # draw the midpoints on the image
            trait_img = cv2.circle(orig, (int(tltrX), int(tltrY)), 15, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, (int(blbrX), int(blbrY)), 15, (255, 0, 0), -1)

            # draw lines between the midpoints
            trait_img = cv2.line(orig, (int(x), int(y+h*0.5)), (int(x+w), int(y+h*0.5)), (255, 0, 255), 6)
            trait_img = cv2.line(orig, (int(x+w*0.5), int(y)), (int(x+w*0.5), int(y+h)), (255, 0, 255), 6)
            

            hull = cv2.convexHull(c)
            
            # draw convexhull in red color
            trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255), 2)
            
            area_c_cmax = cv2.contourArea(c)
            #hull_area = cv2.contourArea(hull)
            
            cnt_area[index] = (area_c_cmax)
            cnt_width.append(w)
            cnt_height.append(h)
            
            
            print("Contour {0} shape info: Width = {1:.2f}, height= {2:.2f}, area = {3:.2f}\n".format(index+1, w, h, area_c_cmax))
   
            
    return trait_img, cnt_area, cnt_width, cnt_height
    



# convert RGB value to HEX format
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))



# get the color pallate
def get_cmap(n, name = 'hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    
    
# dorminant color clustering method 
def color_region(image, mask, save_path, num_clusters):
    
    # read the image
     #grab image width and height
    (h, w) = image.shape[:2]

    #apply the mask to get the segmentation of plant
    masked_image_ori = cv2.bitwise_and(image, image, mask = mask)
    
    #define result path for labeled images
    result_img_path = save_path + 'masked.png'
    cv2.imwrite(result_img_path, masked_image_ori)
    
    # convert to RGB
    image_RGB = cv2.cvtColor(masked_image_ori, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_RGB.reshape((-1, 3))
    
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    #num_clusters = 5
    compactness, labels, (centers) = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels_flat = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels_flat]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image_RGB.shape)


    segmented_image_BRG = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    #define result path for labeled images
    result_img_path = save_path + 'clustered.png'
    cv2.imwrite(result_img_path, segmented_image_BRG)


    '''
    fig = plt.figure()
    ax = Axes3D(fig)        
    for label, pix in zip(labels, segmented_image):
        ax.scatter(pix[0], pix[1], pix[2], color = (centers))
            
    result_file = (save_path + base_name + 'color_cluster_distributation.png')
    plt.savefig(result_file)
    '''
    #Show only one chosen cluster 
    #masked_image = np.copy(image)
    masked_image = np.zeros_like(image_RGB)

    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to render
    #cluster = 2

    cmap = get_cmap(num_clusters+1)
    
    #clrs = sns.color_palette('husl', n_colors = num_clusters)  # a list of RGB tuples

    color_conversion = interp1d([0,1],[0,255])


    for cluster in range(num_clusters):

        print("Processing color cluster {0} ...\n".format(cluster))
        #print(clrs[cluster])
        #print(color_conversion(clrs[cluster]))

        masked_image[labels_flat == cluster] = centers[cluster]

        #print(centers[cluster])

        #convert back to original shape
        masked_image_rp = masked_image.reshape(image_RGB.shape)

        #masked_image_BRG = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('maksed.png', masked_image_BRG)

        gray = cv2.cvtColor(masked_image_rp, cv2.COLOR_BGR2GRAY)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        #thresh_cleaned = clear_border(thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #c = max(cnts, key=cv2.contourArea)

        '''
        # compute the center of the contour area and draw a circle representing the center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        result = cv2.putText(masked_image_rp, "#{}".format(cluster + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        '''
        
        
        if not cnts:
            print("findContours is empty")
        else:
            
            # loop over the (unsorted) contours and draw them
            for (i, c) in enumerate(cnts):

                result = cv2.drawContours(masked_image_rp, c, -1, color_conversion(np.random.random(3)), 2)
                #result = cv2.drawContours(masked_image_rp, c, -1, color_conversion(clrs[cluster]), 2)

            #result = result(np.where(result == 0)== 255)
            result[result == 0] = 255


            result_BRG = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            result_img_path = save_path + 'result_' + str(cluster) + '.png'
            cv2.imwrite(result_img_path, result_BRG)


    
    counts = Counter(labels_flat)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = centers

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    #print(hex_colors)
    
    index_bkg = [index for index in range(len(hex_colors)) if hex_colors[index] == '#000000']
    
    #print(index_bkg[0])

    #print(counts)
    #remove background color 
    del hex_colors[index_bkg[0]]
    del rgb_colors[index_bkg[0]]
    
    # Using dictionary comprehension to find list 
    # keys having value . 
    delete = [key for key in counts if key == index_bkg[0]] 
  
    # delete the key 
    for key in delete: del counts[key] 
   
    fig = plt.figure(figsize = (6, 6))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    #define result path for labeled images
    result_img_path = save_path + 'pie_color.png'
    plt.savefig(result_img_path)

   
    return rgb_colors, counts, hex_colors



# Read barcode in the imageand decode barcode info
def barcode_detect(img_ori):
    
    height, width = img_ori.shape[:2]
    
    barcode_info = decode((img_ori.tobytes(), width, height))
    
    
    if len(barcode_info) > 0:
        
        
        barcode_str = str(barcode_info[0].data)
        
        #print('Decoded data:', barcode_str)
        #print(decoded_object.rect.top, decoded_object.rect.left)
        #print(decoded_object.rect.width, decoded_object.rect.height)
 
        tag_info = re.findall(r"'(.*?)'", barcode_str, re.DOTALL)
        
        tag_info = " ".join(str(x) for x in tag_info)
        
        tag_info = tag_info.replace("'", "")
        
        print("Tag info: {}\n".format(tag_info))
    
    else:
        
        print("barcode information was not readable!\n")
        tag_info = 'Unreadable'
        
    return tag_info
    
    


# Detect marker in the image
def marker_detect(img_ori, template, method, selection_threshold):
    
    # load the image, clone it for output
    img_rgb = img_ori.copy()
      
    # Convert it to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
      
    # Store width and height of template in w and h 
    w, h = template.shape[::-1] 
      
    # Perform match operations. 
    #res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
    
    #res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
    
    
    
    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    
    
    # Specify a threshold for template detection, can be adjusted
    #threshold = 0.8
      
    # Store the coordinates of matched area in a numpy array 
    #loc = np.where( res <= selection_threshold)
    
    loc = np.where( res >= selection_threshold)   
    
    if len(loc):
    
        (y,x) = np.unravel_index(res.argmax(), res.shape)
    
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)

        (startX, startY) = max_loc
        endX = startX + template.shape[1]
        endY = startY + template.shape[0]
        
        
        marker_img = img_ori[startY:endY, startX:endX]
        
        marker_overlay = marker_img
        
        
        
        # Draw a rectangle around the matched region. 
        #for pt in zip(*loc[::-1]): 
            
            #marker_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)

    
        # load the marker image, convert it to grayscale
        marker_img_gray = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY) 
    
       
        # load the image and perform pyramid mean shift filtering to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(marker_img, 21, 51)
        
        # convert the mean shift image to grayscale, then apply Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = imutils.grab_contours(cnts)
        
        largest_cnt = max(cnts, key=cv2.contourArea)
        
        #print("[INFO] {} unique contours found in marker_img\n".format(len(cnts)))
        
        # compute the radius of the detected coin
        # calculate the center of the contour
        M = cv2.moments(largest_cnt )
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        # calculate the radius of the contour from area (I suppose it's a circle)
        area = cv2.contourArea(largest_cnt)
        radius = np.sqrt(area/math.pi)
        coins_width_contour = 2* radius
    
        # draw a circle enclosing the object
        ((x, y), radius) = cv2.minEnclosingCircle(largest_cnt) 
        coins_width_circle = 2* radius
        

    return  marker_img, thresh, coins_width_contour, coins_width_circle



# compute rect region based on left top corner coordinates and dimension of the region
def region_extracted(orig, x, y, w, h):
    
    roi = orig[y:y+h, x:x+w]
    
    return roi


# compute all the traits based on input image
def extract_traits(image_file):

    
    # extarct path and name of the image file
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    file_size = os.path.getsize(image_file)/MBFACTOR
    
    image_file_name = Path(image_file).name


    print("Exacting traits for image : {0}\n".format(str(image_file_name)))
     
    # create result folder
    if (args['result']):
        save_path = args['result']
    else:
        mkpath = os.path.dirname(abs_path) +'/' + base_name
        mkdir(mkpath)
        save_path = mkpath + '/'
        

    print ("results_folder:" + save_path +'\n')
    

    # initialize all the traits output 
    area = kernel_area_ratio = max_width = max_height = avg_curv = n_leaves = 0
    
    if (file_size > 5.0):
        print("It will take some time due to larger file size {0} MB\n".format(str(int(file_size))))
    else:
        print("Segmentaing plant object using automatic color clustering method...\n")
    
    image = cv2.imread(image_file)
    
    #make backup image
    orig = image.copy()
    
    img_height, img_width, img_channels = orig.shape
    
    source_image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    
    # segment mutiple objects in image using thresh method to accquire internal contours
    (left_img, right_img, mask_seg, img_overlay, cnt_area_internal) = mutilple_objects_seg(orig)
    
    # save segmentation result
    result_file = (save_path + base_name + '_overlay' + file_extension)
    cv2.imwrite(result_file, img_overlay)
    
     
    args_colorspace = args['color_space']
    args_channels = args['channels']
    args_num_clusters = args['num_clusters']
    
    #color clustering based object segmentation to accquire external contours
    thresh = color_cluster_seg(image.copy(), args_colorspace, args_channels, args_num_clusters)
    
    
   ###############################################################################################
    #combine external contours and internal contoures to compute object mask
    combined_mask = mask_seg | thresh
    
    
    # Taking a matrix of size 25 as the kernel
    kernel = np.ones((25,25), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    result_file = (save_path + base_name + '_combined_mask' + file_extension)
    #print(filename)
    cv2.imwrite(result_file, combined_mask)
    
    
    thresh_combined_mask = cv2.threshold(combined_mask, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh_combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts_sorted[0:2]:

        # draw the contour and center of the shape on the image
        img_overlay = cv2.drawContours(img_overlay, [c], -1, (0, 255, 0), 2)
    
    # save segmentation result
    result_file = (save_path + base_name + '_overlay_combined' + file_extension)
    cv2.imwrite(result_file, img_overlay)
    
    ################################################################################################################################
    #compute external traits
    (trait_img, cnt_area_external, cnt_width, cnt_height) = comp_external_contour(image.copy(), thresh_combined_mask, img_overlay)
    
    # save segmentation result
    result_file = (save_path + base_name + '_excontour' + file_extension)
    #print(filename)
    cv2.imwrite(result_file, trait_img)
    

    
    #################################################################################################################################
    #compute the area ratio of interal contour verse external contour, kernal area ratio
    area_max_external = max(cnt_area_external)
    
    sum_area_internal = sum(cnt_area_internal)
    

    if (cnt_area_internal[0] >= cnt_area_external[0])  or  (cnt_area_internal[1] >= cnt_area_external[1]):
        
        if sum_area_internal < area_max_external:
            
            cnt_area_external[0] = cnt_area_external[1] = np.mean(cnt_area_external)
        else:
            cnt_area_external[0] = cnt_area_external[1] = area_max_external

    
    area_ratio = (cnt_area_internal[0]/cnt_area_external[0], cnt_area_internal[1]/cnt_area_external[1])
    
    
    area_ratio  = [ elem for elem in area_ratio if elem < 1]
    
    '''
    print("[INFO]: The two kernel_area = {}".format(area_ratio))
    print("[INFO]: The two width = {}".format(cnt_width))
    print("[INFO]: The two height = {}".format(cnt_height))
    '''
    # compute traits needed
    kernel_area_ratio = np.mean(area_ratio)
    kernel_area = sum(cnt_area_internal) / len(cnt_area_internal)
    max_width = sum(cnt_width) / len(cnt_width)
    max_height = sum(cnt_height) / len(cnt_height)
    
    
    
    ###################################################################################################
    # detect coin and bracode uisng template mathcing method
    
    # define right bottom area for coin detection
    x = int(img_width*0.5)
    y = int(img_height*0.5)
    w = int(img_width*0.5)
    h = int(img_height*0.5)
    
    # method for template matching 
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
     
    # detect the coin object based on the template image
    (marker_coin_img, thresh_coin, coins_width_contour, coins_width_circle) = marker_detect(region_extracted(orig, x, y, w, h), tp_coin, methods[0], 0.8)
    
    # save segmentation result
    result_file = (save_path + base_name + '_coin' + file_extension)
    cv2.imwrite(result_file, marker_coin_img)
    
    # Brazil 1 Real coin dimension is 27 × 27 mm
    print("The width of Brazil 1 Real coin in the marker image is {:.0f} × {:.0f} pixels\n".format(coins_width_contour, coins_width_circle))
    

    # define left bottom area for barcode detection
    x = 0
    y = int(img_height*0.5)
    w = int(img_width*0.5)
    h = int(img_height*0.5)
    
    # detect the barcode object based on template image
    (marker_barcode_img, thresh_barcode, barcode_width_contour, barcode_width_circle) = marker_detect(region_extracted(orig, x, y, w, h), tp_barcode, methods[0], 0.8)
    
    # save segmentation result
    result_file = (save_path + base_name + '_barcode' + file_extension)
    cv2.imwrite(result_file, marker_barcode_img)
    
    # parse barcode image using pylibdmtx lib
    tag_info = barcode_detect(marker_barcode_img)
    

    '''
    ################################################################################################
    # analyze color distribution of the segmented objects
    
    num_clusters = 5
    
    #save color analysisi/quantization result
    (rgb_colors, counts, hex_colors) = color_region(image.copy(), thresh_combined_mask, save_path, num_clusters)
    

    #print("hex_colors = {} {}\n".format(hex_colors, type(hex_colors)))
    
    list_counts = list(counts.values())
    
    #list_hex_colors = list(hex_colors)
    
    #print(type(list_counts))
    
    color_ratio = []
    
    for value_counts, value_hex in zip(list_counts, hex_colors):
        
        #print(percentage(value, np.sum(list_counts)))
        
        color_ratio.append(percentage(value_counts, np.sum(list_counts)))

    
    ####################################################
    '''
      
    
    #return image_file_name, tag_info, kernel_area, kernel_area_ratio, max_width, max_height, color_ratio, hex_colors
    
    return image_file_name, tag_info, kernel_area, kernel_area_ratio, max_width, max_height
    





if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = True,    help = "Image filetype")
    ap.add_argument('-mk', '--marker', required = False,  default ='/marker_template/coin.png',  help = "Marker file name")
    ap.add_argument('-bc', '--barcode', required = False,  default ='/marker_template/barcode.png',  help = "Barcode file name")
    ap.add_argument("-r", "--result", required = False,    help="result path")
    ap.add_argument('-s', '--color-space', type = str, required = False, default ='lab', help='Color space to use: BGR , HSV, Lab(default), YCrCb (YCC)')
    ap.add_argument('-c', '--channels', type = str, required = False, default='1', help='Channel indices to use for clustering, where 0 is the first channel,' 
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" ' 
                                                                       + 'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type = int, required = False, default = 2,  help = 'Number of clusters for K-means clustering (default 2, min 2).')
    ap.add_argument('-min', '--min_size', type = int, required = False, default = 250000,  help = 'min size of object to be segmented.')
    
    args = vars(ap.parse_args())
    
    
    # parse input arguments
    file_path = args["path"]
    ext = args['filetype']
    
    coin_path = args["marker"]
    barcode_path = args["barcode"]
    
    min_size = args['min_size']
    #min_distance_value = args['min_dist']
    
    
    # path of the marker (coin), default path will be '/marker_template/marker.png' and '/marker_template/barcode.png'
    # can be changed based on requirement
    global  tp_coin, tp_barcode

    
    #setup marker path to load template
    template_path = file_path + coin_path
    barcode_path = file_path + barcode_path
    
    try:
        # check to see if file is readable
        with open(template_path) as tempFile:

            # Read the template 
            tp_coin = cv2.imread(template_path, 0)
            print("Template loaded successfully...")
            
    except IOError as err:
        
        print("Error reading the Template file {0}: {1}".format(template_path, err))
        exit(0)

    
    try:
        # check to see if file is readable
        with open(barcode_path) as tempFile:

            # Read the template 
            tp_barcode = cv2.imread(barcode_path, 0)
            print("Barcode loaded successfully...\n")
            
    except IOError as err:
        
        print("Error reading the Barcode file {0}: {1}".format(barcode_path, err))
        exit(0)
    
    

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    #print((imgList))
    
    n_images = len(imgList)
    
    result_list = []

    '''
    
    #loop execute
    for image in imgList:
        
        #(filename, tag_info, kernel_area, kernel_area_ratio, max_width, max_height, color_ratio, hex_colors) = extract_traits(image)
        #result_list.append([filename, tag_info, kernel_area, kernel_area_ratio, max_width, max_height, color_ratio[0], color_ratio[1], color_ratio[2], color_ratio[3], hex_colors[0], hex_colors[1], hex_colors[2], hex_colors[3]])
        
        (filename, tag_info, kernel_area, kernel_area_ratio, max_width, max_height) = extract_traits(image)

        result_list.append([filename, tag_info, kernel_area, kernel_area_ratio, max_width, max_height])

    '''
    # get cpu number for parallel processing
    agents = psutil.cpu_count()   
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(extract_traits, imgList)
        pool.terminate()
    
    
    filename = list(zip(*result))[0]
    tag_info = list(zip(*result))[1]
    avg_kernel_area = list(zip(*result))[2]
    avg_kernel_area_ratio = list(zip(*result))[3]
    avg_width = list(zip(*result))[4]
    avg_height = list(zip(*result))[5]

    
    for i, (v0,v1,v2,v3,v4,v5) in enumerate(zip(filename, tag_info, avg_kernel_area, avg_kernel_area_ratio, avg_width, avg_height)):

        result_list.append([v0,v1,v2,v3,v4,v5])
    
    
    
    #trait_file = (os.path.dirname(os.path.abspath(file_path)) + '/' + 'trait.xlsx')
    
    print("Summary: {0} plant images were processed...\n".format(n_images))
    
    #output in command window in a sum table
 
    #table = tabulate(result_list, headers = ['filename', 'mazie_ear_area', 'kernel_area_ratio', 'max_width', 'max_height' ,'cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 1 hex value', 'cluster 2 hex value', 'cluster 3 hex value', 'cluster 4 hex value'], tablefmt = 'orgtbl')
    
    table = tabulate(result_list, headers = ['filename', 'tag_info', 'avg_kernel_area', 'avg_kernel_area_ratio', 'avg_width', 'avg_height'], tablefmt = 'orgtbl')
    
    print(table + "\n")
    
    

    
    if (args['result']):

        trait_file = (args['result'] + 'trait.xlsx')
        trait_file_csv = (args['result'] + 'trait.csv')
    else:
        trait_file = (file_path + 'trait.xlsx')
        trait_file_csv = (file_path + 'trait.csv')
    
    
    if os.path.isfile(trait_file):
        # update values
        #Open an xlsx for reading
        wb = openpyxl.load_workbook(trait_file)

        #Get the current Active Sheet
        sheet = wb.active
        
        sheet.delete_rows(2, sheet.max_row+1) # for entire sheet
        
        #sheet_leaf = wb.create_sheet()

    else:
        # Keep presets
        wb = openpyxl.Workbook()
        sheet = wb.active
        
        sheet_leaf = wb.create_sheet()

        sheet.cell(row = 1, column = 1).value = 'filename'
        sheet.cell(row = 1, column = 2).value = 'tag_info'
        sheet.cell(row = 1, column = 3).value = 'avg_kernel_area'
        sheet.cell(row = 1, column = 4).value = 'avg_kernel_area_ratio'
        sheet.cell(row = 1, column = 5).value = 'avg_width'
        sheet.cell(row = 1, column = 6).value = 'avg_height'

        '''
        sheet.cell(row = 1, column = 7).value = 'color distribution cluster 1'
        sheet.cell(row = 1, column = 8).value = 'color distribution cluster 2'
        sheet.cell(row = 1, column = 9).value = 'color distribution cluster 3'
        sheet.cell(row = 1, column = 10).value = 'color distribution cluster 4'
        sheet.cell(row = 1, column = 11).value = 'color cluster 1 hex value'
        sheet.cell(row = 1, column = 12).value = 'color cluster 2 hex value'
        sheet.cell(row = 1, column = 13).value = 'color cluster 3 hex value'
        sheet.cell(row = 1, column = 14).value = 'color cluster 4 hex value'        
        '''

    for row in result_list:
        sheet.append(row)
   
    
    #save the csv file
    wb.save(trait_file)
    
    
    


    

    
