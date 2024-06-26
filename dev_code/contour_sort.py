"""
Version: 1.5

Summary: compute the segmentaiton and label of cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 contour_sort.py -p ~/example/Tara_data/color_checker/  -ft jpg


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")

"""


# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
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
        
        
def image_label(image_file):
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    #Parse image path  and create result image path
    path, filename = os.path.split(image_file)

    print("processing image : {0} \n".format(str(filename)))

    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    image = cv2.imread(image_file)
    
    orig = image.copy()
    
    img_width, img_height, img_channels = image.shape

     
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    
    orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(orig_hsv, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    

####################################################################################
    
    # apply connected component analysis to the thresholded image
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)


    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        area = stats[i, cv2.CC_STAT_AREA]
        
        (cX, cY) = centroids[i]
        
        if area > 1500 and area < 200000:
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid

            cp_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cp_image = cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        

    #define result path for labeled images
    result_img_path = save_path_ac + str(filename[0:-4]) + '_cp.jpg'
    cv2.imwrite(result_img_path, cp_image)
    
    
    '''
    thresh = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    
    thresh_dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    thresh_erosion = cv2.erode(thresh, kernel, iterations=1)
    
    #define result path for labeled images
    result_img_path = save_path_label + str(filename[0:-4]) + '_thresh.jpg'
    cv2.imwrite(result_img_path, thresh_erosion)
    '''
    '''
    ##############################################################################################
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    
    for c in cnts_sorted:
        
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        
        # draw the contour and center of the shape on the image
        contour_img = cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        contour_img = cv2.circle(image, (cX, cY), 14, (0, 0, 255), -1)
        contour_img = cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    #define result path for labeled images
    result_img_path = save_path_ac + str(filename[0:-4]) + '_contour.jpg'
    cv2.imwrite(result_img_path, contour_img)
    '''

    
    
    



if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")
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
    mkpath = file_path + str('active_component')
    mkdir(mkpath)
    save_path_ac = mkpath + '/'
    
    mkpath = file_path + str('lable')
    mkdir(mkpath)
    save_path_label = mkpath + '/'

    #print "results_folder: " + save_path_ac  
    
    # Run this with a pool of avaliable agents having a chunksize of 3 until finished
    # run image labeling fucntion to accquire segmentation for each cross section image
    agents = multiprocessing.cpu_count() - 2
    chunksize = 3
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(image_label, imgList, chunksize)
        pool.terminate()
        
    
    
    
    
    

    
