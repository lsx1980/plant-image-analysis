"""
Version: 1.0

Summary: RGB to IR

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 RGB_LAB.py -p ~/example/ -ft jpg


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")

"""

import os
import glob
import sys
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mtpltcm
import shutil
import argparse

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
        print (path + ' folder constructed!')
        # make dir
        os.makedirs(path)
        return True
    else:
       # if exists, return 
        print (path +' path exists!\n')
        shutil.rmtree(path)
        os.makedirs(path)
        return False


def RGB_LAB(image_file):
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    #initialize the colormap
    colormap = mpl.cm.jet
    cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
    scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)
    
    # load original image
    image = cv2.imread(image_file)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # Convert color space to LAB space and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    

    #add blur to make it more realistic
    #blur = cv2.GaussianBlur(gray,(1,1),0)
    
    #assign colormap
    colors = scalarMap.to_rgba(gray, bytes=False)
    
    # save result
    result_file = (save_path + base_name + file_extension)
    
   
    #assign colormap
    colors_L = scalarMap.to_rgba(L.copy(), bytes=False)
    
    colors_L = cv2.convertScaleAbs(colors_L.copy(), alpha=(255.0))
    
    colors_L_gray = cv2.cvtColor(colors_L, cv2.COLOR_BGR2GRAY)
    
    # save Lab result
    result_file = (save_path + base_name + file_extension)
    cv2.imwrite(result_file, colors_L_gray)
    

    
    


if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")
    args = vars(ap.parse_args())

    # setting path to cross section image files
    file_path = args["path"]
    ext = args['filetype']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
     # make the folder to store the results
    #parent_path = os.path.abspath(os.path.join(file_path, os.pardir))
    mkpath = file_path + '/' + str('output')
    mkdir(mkpath)
    save_path = mkpath + '/'
    

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    for idx, image_file in enumerate(imgList):
        
        RGB_LAB(image_file)
        
        
    
    


