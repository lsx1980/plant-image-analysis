'''
Name: circle_detection.py

Version: 1.0

Summary: Detect circle shape markers in image and cropp image based on marker location
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2021-03-09

USAGE:

time python3 marker_roi_crop.py -p ~/plant-image-analysis/test/ -ft png 

'''

# import necessary packages
import argparse
import cv2
import numpy as np
import os
import glob
from pathlib import Path 

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

from tabulate import tabulate
import openpyxl

from pathlib import Path


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

# Detect circles in the image
def circle_detect(image_file):
    
    image_file_name = Path(image_file).name
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    print("Processing image : {0}\n".format(str(image_file)))
     
    # save folder construction
    mkpath = os.path.dirname(abs_path) +'/cropped'
    mkdir(mkpath)
    save_path = mkpath + '/'

    print ("results_folder: " + save_path)
    
   

    # load the image, clone it for output, and then convert it to grayscale
    img_ori = cv2.imread(image_file)
    
    img_rgb = img_ori.copy()
      
    # Convert it to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
      
    # Store width and height of template in w and h 
    w, h = template.shape[::-1] 
      
    # Perform match operations. 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) 
    
    # Specify a threshold 
    threshold = 0.8
      
    # Store the coordinates of matched area in a numpy array 
    loc = np.where( res >= threshold)  
    
    if len(loc):
    
        (y,x) = np.unravel_index(res.argmax(), res.shape)
    
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)
    
        print(y,x)
        
        print(min_val, max_val, min_loc, max_loc)
    
        # Draw a rectangle around the matched region. 
        for pt in zip(*loc[::-1]): 
            circle_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 

        # save segmentation result
        #result_file = (save_path + base_name + '_circle.' + args['filetype'])
        #print(result_file)
        #cv2.imwrite(result_file, circle_overlay)
        
        #crop_img = img_rgb[y-200:y+950, x-950:x+450]
        
        crop_img = img_rgb[y+150:y+750, x-650:x]
        # save segmentation result
        result_file = (save_path + base_name + '_cropped.' + args['filetype'])
        print(result_file)
        cv2.imwrite(result_file, crop_img)

    return image_file_name, (x,y)





if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help = "image filetype")

    args = vars(ap.parse_args())
    
    
    # Setting path to image files
    file_path = args["path"]
    ext = args['filetype']

    # Extract file type and path
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    # Accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    global  template
    template_path = "/home/suxing/plant-image-analysis/marker_template/marker_rotate.png"
    # Read the template 
    template = cv2.imread(template_path, 0) 
    print(template)
    #imgList = (glob.glob(image_file_path))

    #print((imgList))
    #global save_path
    
    # Get number of images in the data folder
    n_images = len(imgList)
    
    result_list = []
    
    
    # Loop execute
    for image in imgList:
        
        (image_file_name, circle_overlay) = circle_detect(image)
        
        result_list.append([image_file_name, circle_overlay])
        
        circle_detect(image)
   
    '''
    # Parallel processing
    
    # get cpu number for parallel processing
    agents = psutil.cpu_count()   
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result_list = pool.map(circle_detect, imgList)
        pool.terminate()
    '''
    
    # Output sum table in command window
    print("Summary: {0} plant images were processed...\n".format(n_images))
    
    table = tabulate(result_list, headers = ['image_file_name', 'marker coordinates'], tablefmt = 'orgtbl')

    print(table + "\n")
    

