"""
Version: 1.5

Summary: Automatic image brightness adjustment based on gamma correction method

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 gamma_correction.py -p ~/plant-image-analysis/test/ -ft jpg 

argument:
("-p", "--path", required = True,    help="path to image file")
("-ft", "--filetype", required=True,    help="Image filetype") 

"""

#!/usr/bin/python
# Standard Libraries

import os,fnmatch
import argparse
import shutil
import cv2

import numpy as np

import glob


import multiprocessing
from multiprocessing import Pool
from contextlib import closing

import resource

from PIL import Image, ImageEnhance


# create result folder
def mkdir(path):
    # import module
    #import os
 
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


#adjust the gamma value to increase the brightness of image
def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to perfrom image enhancement
def image_enhance(img):

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  
    
    # split on 3 different channels
    l, a, b = cv2.split(lab)  

    # apply CLAHE to the L-channel
    l2 = clahe.apply(l)  

    # merge channels
    lab = cv2.merge((l2,a,b))  
    
    # convert from LAB to BGR
    img_enhance = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  
    
    return img_enhance

# Convert it to LAB color space to access the luminous channel which is independent of colors.
def isbright(image_file):
    
    # Set up threshold value for luminous channel, can be adjusted and generalized 
    thresh = 0.1
    
    # Load image file 
    orig = cv2.imread(image_file)
    
    # Make backup image
    image = orig.copy()
    
    # Get file name
    #abs_path = os.path.abspath(image_file)
    
    #filename, file_extension = os.path.splitext(abs_path)
    #base_name = os.path.splitext(os.path.basename(filename))[0]

    #image_file_name = Path(image_file).name
    
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    
    text_bool = "bright" if np.mean(L) < thresh else "dark"
    
    print(np.mean(L))
    
    return np.mean(L) > thresh
    

def gamma_correction(image_file):
    
  
    #parse the file name 
    path, filename = os.path.split(image_file)
    
    #filename, file_extension = os.path.splitext(image_file)
    
    # construct the result file path
    result_img_path = save_path + str(filename[0:-4]) + '.' + ext
    
    print("Enhancing image : {0} \n".format(str(filename)))
    
    # Load the image
    image = cv2.imread(image_file)
    
    #get size of image
    img_height, img_width = image.shape[:2]
    
    #image = cv2.resize(image, (0,0), fx = scale_factor, fy = scale_factor) 
    
    gamma = args['gamma']
    
    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    
    adjusted = adjust_gamma(image, gamma=gamma)
    
    enhanced_image = image_enhance(adjusted)
    
    

    # save result as images for reference
    cv2.imwrite(result_img_path,enhanced_image)


def image_enhance(image_file):
    
    #parse the file name 
    path, filename = os.path.split(image_file)
    
    #filename, file_extension = os.path.splitext(image_file)
    
    # construct the result file path
    result_img_path = save_path + str(filename[0:-4]) + '.' + ext
    
    print("Enhancing image : {0} \n".format(str(filename)))
    
    im = Image.open(image_file)
    
    im_sharpness = ImageEnhance.Sharpness(im).enhance(1.5)
    
    im_contrast = ImageEnhance.Contrast(im_sharpness).enhance(1.5)

    im_out = ImageEnhance.Brightness(im_contrast).enhance(0.8)
    
    im_out.save(result_img_path)
    

def Adaptive_Histogram_Equalization(image_file):
    
     #parse the file name 
    path, filename = os.path.split(image_file)
    
    #filename, file_extension = os.path.splitext(image_file)
    
    # construct the result file path
    result_img_path = save_path + str(filename[0:-4]) + '.' + ext
    
    print("AHE image : {0} \n".format(str(filename)))

    #print(isbright(image_file))

     # Load the image
    bgr = cv2.imread(image_file)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # save result as images for reference
    cv2.imwrite(result_img_path, out)


if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False,  default = 'jpg',  help = "image filetype")
    ap.add_argument("-gamma", "--gamma", type = float, required = False,  default = 0.5,  help = "gamma value")
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


    # make the folder to store the results
    parent_path = os.path.abspath(os.path.join(file_path, os.pardir))
    #mkpath = parent_path + '/' + str('gamma_correction')
    mkpath = file_path + '/' + str('gamma_correction')
    mkdir(mkpath)
    save_path = mkpath + '/'

    #print "results_folder: " + save_path  
    
    # Loop execute
    for image in imgList:
        
        Adaptive_Histogram_Equalization(image)
        

    '''

    
    # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()-1
    

    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(Adaptive_Histogram_Equalization, imgList)
        pool.terminate()
    '''
      
    # monitor memory usage 
    rusage_denom = 1024.0
    
    print("Memory usage: {0} MB\n".format(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom)))
    

    

    





    

    

    
  

   
    
    




