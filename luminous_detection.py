'''
Name: luminous_detection.py

Version: 1.0

Summary: Detect dark images by converting it to LAB color space to access the luminous channel which is independent of colors.
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2021-03-09

USAGE:

time python3 luminous_detection.py -p ~/plant-image-analysis/test/ -ft png 

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

# Convert it to LAB color space to access the luminous channel which is independent of colors.
def isbright(image_file):
    
    # Set up threshold value for luminous channel, can be adjusted and generalized 
    thresh = 0.5
    
    # Load image file 
    orig = cv2.imread(image_file)
    
    # Make backup image
    image = orig.copy()
    
    # Get file name
    #abs_path = os.path.abspath(image_file)
    
    #filename, file_extension = os.path.splitext(abs_path)
    #base_name = os.path.splitext(os.path.basename(filename))[0]

    image_file_name = Path(image_file).name
    
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    
    text_bool = "bright" if np.mean(L) < thresh else "dark"

    return image_file_name, np.mean(L), text_bool
    

def result_excel(result_list, file_path):
    
    result_file = (file_path + '/luminous_detection.xlsx')
    
    print(result_file)
    
    if os.path.isfile(result_file):
        # update values
        #Open an xlsx for reading
        wb = openpyxl.load_workbook(result_file)

        #Get the current Active Sheet
        sheet = wb.active

    else:
        # Keep presets
        wb = openpyxl.Workbook()
        sheet = wb.active

        sheet.cell(row = 1, column = 1).value = 'image_file_name'
        sheet.cell(row = 1, column = 2).value = 'luminous_avg'
        sheet.cell(row = 1, column = 3).value = 'dark_or_bright'
    
    for row in result_list:
        sheet.append(row)
    
    #save the csv file
    wb.save(result_file)
    

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

    #print((imgList))
    #global save_path
    
    # Get number of images in the data folder
    n_images = len(imgList)
    
    result_list = []
    
    '''
    # Loop execute
    for image in imgList:
        
        (image_file_name, luminous_avg, dark_or_bright) = isbright(image)
        
        result_list.append([image_file_name, luminous_avg, dark_or_bright])
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
        result_list = pool.map(isbright, imgList)
        pool.terminate()
    

    # Output sum table in command window 
    print("Summary: {0} plant images were processed...\n".format(n_images))
    
    table = tabulate(result_list, headers = ['image_file_name', 'luminous_avg', 'dark_or_bright'], tablefmt = 'orgtbl')

    print(table + "\n")
    
    result_excel(result_list, file_path)

