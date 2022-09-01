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





def sort_contours(cnts, method = "left-to-right"):
    
    """sort contours based on user defined method
    
    Inputs: 
    
        cnts: contours extracted from mask image
        
        method: user defined method, default was "left-to-right"
        

    Returns:
    
        sorted_cnts: list of sorted contours 
        
    """   
    
    
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
        
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    (sorted_cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    
    # return the list of sorted contours and bounding boxes
    return sorted_cnts





def adaptive_threshold_external(image_file):
    
    """compute thresh image using adaptive threshold Method
    
    Inputs: 
    
        maksed_img: masked image contains only target objects
        
        GaussianBlur_ksize: Gaussian Kernel Size 
        
        blockSize: size of the pixelneighborhood used to calculate the threshold value
        
        weighted_mean: the constant used in the both methods (subtracted from the mean or weighted mean).

    Returns:
        
        thresh_adaptive_threshold: thresh image using adaptive thrshold Method
        
        maksed_img_adaptive_threshold: masked image using thresh_adaptive_threshold

    """
    
    # set the parameters for adoptive threshholding method
    GaussianBlur_ksize = 5
    
    blockSize = 41
    
    weighted_mean = 10
        
    # load the image 
    #Parse image path  and create result image path
    path, filename = os.path.split(image_file)

    print("processing image : {0} \n".format(str(filename)))

    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    img = cv2.imread(image_file)
    
    # obtain image dimension
    img_height, img_width, n_channels = img.shape
    
    orig = img.copy()
    
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blurring it . Applying Gaussian blurring with a GaussianBlur_ksize×GaussianBlur_ksize kernel 
    # helps remove some of the high frequency edges in the image that we are not concerned with and allow us to obtain a more “clean” segmentation.
    blurred = cv2.GaussianBlur(gray, (GaussianBlur_ksize, GaussianBlur_ksize), 0)

    # adaptive method to be used. 'ADAPTIVE_THRESH_MEAN_C' or 'ADAPTIVE_THRESH_GAUSSIAN_C'
    thresh_adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10)

    # apply individual object mask
    maksed_img_adaptive_threshold = cv2.bitwise_and(orig, orig.copy(), mask = ~thresh_adaptive_threshold)

    
    #find contours and get the external one
    contours, hier = cv2.findContours(thresh_adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort the contours based on area from largest to smallest
    contours_sorted = sorted(contours, key = cv2.contourArea, reverse = True)
    
    #contours_sorted = contours
    
    
    rect_area_rec = []
    
    for index, c in enumerate(contours_sorted):
        
        #get the bounding rect
            (x, y, w, h) = cv2.boundingRect(c)
            
            rect_area_rec.append(w*h)
    
    idx_sort = [i[0] for i in sorted(enumerate(rect_area_rec), key=lambda k: k[1], reverse=True)]
    
    
    rect_center_rec = []
    
    rect_width_rec = []
    
    for index, value in enumerate(idx_sort[0:3]):
        
        c = contours_sorted[value]
        
        #get the bounding rect
        (x, y, w, h) = cv2.boundingRect(c)
        
        center = (x, y)
        rect_center_rec.append(center)
        rect_width_rec.append(w)

    from scipy.spatial.distance import pdist
    
    print(pdist(rect_center_rec))
    
    mindist = np.min(pdist(rect_center_rec))
    
    
    
    #print(idx_sort)
    #finding closest point among the center list of the circles to the right-bottom of the image
    #idx_closest = closest_center((0 + img_width, 0 + img_height), circle_center_coord)
    
    area_rec = []
    
    trait_img = orig
    
    for index, value in enumerate(idx_sort):
        
        if index < 2:
             
            # visualize only the two external contours and its bounding box
            c = contours_sorted[value]
            
             # compute the convex hull of the contour
            hull = cv2.convexHull(c)
            
            hullArea = float(cv2.contourArea(hull))
            
            area_rec.append(hullArea)
            
            #get the bounding rect
            (x, y, w, h) = cv2.boundingRect(c)
            
            # draw a rectangle to visualize the bounding rect
            #trait_img = cv2.drawContours(orig, c, -1, (255, 255, 0), 3)
            
            #area_c_cmax = cv2.contourArea(c)
            
            trait_img = cv2.putText(orig, "#{0}, area = {1}".format(index,hullArea), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            
            # draw a green rectangle to visualize the bounding rect
            trait_img = cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 4)
            
            # draw convexhull in red color
            trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255), 4)
            

    external_contour_area = sum(area_rec)/len(area_rec)
    
    #print(external_contour_area)
       
    #define result path for labeled images
    #result_img_path = save_path + str(filename[0:-4]) + '_thresh.png'
    
    # save results
    #cv2.imwrite(result_img_path, thresh_adaptive_threshold)
    
    #define result path for labeled images
    result_img_path = save_path + str(filename[0:-4]) + '_ctr.png'
    
    # save results
    cv2.imwrite(result_img_path, trait_img)
    
    



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
        result = pool.map(adaptive_threshold_external, imgList, chunksize)
        pool.terminate()
        
    '''
    #loop execute to get all traits
    for image in imgList:
        
        print(image)
        circle_detection(image)
        print()
    '''

    
