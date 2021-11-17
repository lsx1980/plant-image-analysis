'''
Name: color_segmentation.py

Version: 1.0

Summary: grid segmentation.  
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2018-05-29

USAGE:

python3 grid_seg.py -p ~/plant-image-analysis/test/ -ft jpg -r 6 -c 5


'''

# import the necessary packages
import os
import glob

import numpy as np
import argparse
import cv2


import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

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
        


def segmentation(image_file):
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(image_file)
    
    file_size = os.path.getsize(image_file)/MBFACTOR
    
    print("Segmenting image : {0} \n".format(str(filename)))
    
    # load original image
    image = cv2.imread(image_file)
    
    img_height, img_width, img_channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # make the folder to store the results
    #current_path = abs_path + '/'
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # save folder construction
    mkpath = os.path.dirname(abs_path) +'/' + base_name
    mkdir(mkpath)
    save_path = mkpath + '/'
    

    print("results_folder: {0}\n".format(str(save_path)))  
    
    if (file_size > 5.0):
        print("It will take some time due to large file size {0} MB".format(str(int(file_size))))
    else:
        print("Segmenting plant image into blocks... ")
    
    #make backup image
    orig = image.copy()
    

    #number of rows
    nRows = args['nRows']
    # Number of columns
    mCols = args['mCols']

    # Dimensions of the image
    sizeX = img_width
    sizeY = img_height
    #print(img.shape)


    for i in range(0, nRows):
        
        for j in range(0, mCols):
            
            roi = orig[int(i*sizeY/nRows):int(i*sizeY/nRows) + int(sizeY/nRows),int(j*sizeX/mCols):int(j*sizeX/mCols) + int(sizeX/mCols)]
            
            result_file = (save_path +  str(i+1) + str(j+1) + '.' + ext)
            
            cv2.imwrite(result_file, roi)
    
    #return thresh
    #trait_img
    
    
    

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--image', required = True, help = 'Path to image file')
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    ap.add_argument("-r", "--nRows", required=True,  type = int,  help="number of rows")
    ap.add_argument("-c", "--mCols", required=True,  type = int,  help="number of columns")
    args = vars(ap.parse_args())
    
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    

    
    print((imgList))
    
    for image in imgList:
        
        segmentation(image)
    
    #current_img = imgList[0]
    
    #(thresh, trait_img) = segmentation(current_img)
    
    '''
     # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()
    print("Using {0} cores to perform parallel processing... \n".format(int(agents)))

    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(segmentation, imgList)
        pool.terminate()
    '''
    

        


    

    

    
