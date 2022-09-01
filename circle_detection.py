"""
Version: 1.5

Summary: compute the segmentaiton and label of cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 circle_detection.py -p ~/example/plant_test/seeds/test/  -ft png


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")

"""


# import the necessary packages

import numpy as np
import argparse
import imutils
import cv2
 
import glob
import os,fnmatch

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

def mkdir(path):
    """Create result folder"""
 
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


# compute rect region based on left top corner coordinates and dimension of the region
def region_extracted(orig, x, y, w, h):
    
    """compute rect region based on left top corner coordinates and dimension of the region
    
    Inputs: 
    
        orig: image
        
        x, y: left top corner coordinates 
        
        w, h: dimension of the region

    Returns:
    
        roi: region of interest
        
    """   
    roi = orig[y:y+h, x:x+w]
    
    return roi



#adjust the gamma value to increase the brightness of image
def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



def circle_detection(image_file):
    # load the image 
    #Parse image path  and create result image path
    path, filename = os.path.split(image_file)

    print("processing image : {0} \n".format(str(filename)))

    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    img = cv2.imread(image_file)
    
    img_height, img_width, img_channels = img.shape

    '''
    # apply gamma correction and show the images
    gamma = 1.5
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(img, gamma=gamma)
    '''

    # define right bottom area for coin detection
    x = int(img_width*0.5)
    y = int(img_height*0.5)
    w = int(img_width*0.5)
    h = int(img_height*0.5)

     
    # detect the coin object based on the template image
    cropped = region_extracted(img, x, y, w, h)
    
    
    # apply gamma correction and show the images
    gamma = 1.5
    gamma = gamma if gamma > 0 else 0.1
    image = adjust_gamma(cropped, gamma=gamma)
    
    
    
    output = image.copy()
    
    circle_detection_img = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.medianBlur(gray, 25)
    
    minDist = 100
    param1 = 30 #500
    param2 = 30 #200 #smaller value-> more false circles
    minRadius = 80
    maxRadius = 120 #10
    
    # detect circles in the image
    #circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.5, 100)
    
    
    
    # ensure at least some circles were found
    if circles is not None:
        
        print("found {} round objects\n".format(len(circles)))
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            circle_detection_img = cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            circle_detection_img = cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    #define result path for labeled images
    result_img_path = save_path + str(filename[0:-4]) + '_circle.png'
    
    # save results
    cv2.imwrite(result_img_path, circle_detection_img)
   
    
    



if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'png',   help = "Image filetype")
    args = vars(ap.parse_args())

    global save_path_ac, save_path_label
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
     
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    # make the folder to store the results
    mkpath = file_path + str('circle_detection')
    mkdir(mkpath)
    save_path = mkpath + '/'
    

    
    
    # Run this with a pool of avaliable agents having a chunksize of 3 until finished
    # run image labeling fucntion to accquire segmentation for each cross section image
    agents = multiprocessing.cpu_count() - 2
    chunksize = 3
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(circle_detection, imgList, chunksize)
        pool.terminate()
        
    '''
    #loop execute to get all traits
    for image in imgList:
        
        print(image)
        circle_detection(image)
        print()
    '''

    
