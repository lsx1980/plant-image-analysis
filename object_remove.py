"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 object_remove.py -p ~/example/molly_HLN_models_skeleton/Models/HighN/ 

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib
import os
import glob
import shutil
from pathlib import Path
from scipy.spatial import KDTree
import cv2
import imutils

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing




# generate folder to store the output results
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
        print (path+' path exists!')
        return False



#find the closest points from a points sets to a fix point using Kdtree, O(log n) 
def closest_point(point_set, anchor_point):
    
    kdtree = KDTree(point_set)
    
    (d, i) = kdtree.query(anchor_point)
    
    #print("closest point:", point_set[i])
    
    return  (i, point_set[i])
    



def object_remove(image_file):
    
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
   
    image_file_name = Path(image_file).name
   
    
    folder_cur = os.path.basename(abs_path)
    
    mkpath = os.path.dirname(abs_path) + '/plant_result'
    mkdir(mkpath)
    save_path = mkpath + '/'
    
    #print("folder_cur : {0}\n".format(str(folder_cur)))
    
    # make the folder to store the results
    #current_path = abs_path + '/'
    
    print("Exacting traits for image : {0}\n".format(str(image_file_name)))
    
    # load the image, convert it to grayscale, blur it slightly, and threshold it
    
    image = cv2.imread(image_file)
    
    # get the dimension of the image
    img_height, img_width, img_channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    sorted_contours= sorted(cnts, key=cv2.contourArea, reverse= True)

    largest_item = sorted_contours[0]
    
    img_mask = np.zeros([img_height, img_width], dtype="uint8")

    img_mask = cv2.drawContours(image = img_mask, contours = [largest_item], contourIdx = -1, color = (255,255,255), thickness = cv2.FILLED)

    '''
    ################################################################
    center_img = (int(img_height*0.5), int(img_width*0.5))
    
    point_set = np.zeros((len(cnts), 2))
    
    img_mask = np.zeros([img_height, img_width], dtype="uint8")
    
    if len(cnts) > 0:
        # loop over the contours
        for idx, c in enumerate(cnts):
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            #point_set.append[(cY, cX)]
            
            point_set[idx,0] = cY
            point_set[idx,1] = cX


        (index_cp, value_cp) = closest_point(point_set, center_img)
        
        print(index_cp, value_cp)

        if 0 <= index_cp < len(cnts):
            
            img_mask = cv2.drawContours(image = img_mask, contours = [cnts[index_cp]], contourIdx = -1, color = (255,255,255), thickness = cv2.FILLED)

        else:
            
            print("No contour was found!\n")
    ################################################################################
    '''
    
    # save segmentation result
    result_file = (save_path + base_name + '_seg' + file_extension)
    #print(filename)
    cv2.imwrite(result_file, img_mask)
        
        
    return img_mask


        

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    args = vars(ap.parse_args())
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    #print((imgList))
    #global save_path
    
    n_images = len(imgList)
    
    result_list = []
    

    
    '''
    #loop execute
    for image_id, image in enumerate(imgList):
        

        (filename, QR_data, area, solidity, longest_axis, n_leaves, hex_colors, color_ratio, diff_level_1, diff_level_2, diff_level_3, color_diff_a, color_diff_b, color_diff_c) = extract_traits(image)
        
        
        result_list.append([filename, QR_data, area, solidity, longest_axis, n_leaves, 
                            hex_colors[0], color_ratio[0], diff_level_1[0], diff_level_2[0], diff_level_3[0],
                            hex_colors[1], color_ratio[1], diff_level_1[1], diff_level_2[1], diff_level_3[1],
                            hex_colors[2], color_ratio[2], diff_level_1[2], diff_level_2[2], diff_level_3[2],
                            color_diff_a, color_diff_b, color_diff_c])
    '''
    
    
    ########################################################################
    
    # get cpu number for parallel processing
    agents = psutil.cpu_count()-2
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(object_remove, imgList)
        pool.terminate()
        
    
