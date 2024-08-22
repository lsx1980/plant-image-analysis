"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 batch_file_move.py -p ~/example/plant_test/062724_vis/ -r ~/example/plant_test/062724_vis/

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib
import os
import glob
import shutil

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing


from split_image import split_image

import pathlib


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
        #shutil.rmtree(path)
        #os.makedirs(path)
        return False
        



# execute script inside program
def execute_script(cmd_line):
    
    try:
        #print(cmd_line)
        #os.system(cmd_line)

        process = subprocess.getoutput(cmd_line)
        
        print(process)
        
        #process = subprocess.Popen(cmd_line, shell = True, stdout = subprocess.PIPE)
        #process.wait()
        #print (process.communicate())
        
    except OSError:
        
        print("Failed ...!\n")




# execute pipeline scripts in order
def file_move(source_file_path, target_file_path):
    
    
    batch_cmd = "cp " + source_file_path + " " + target_file_path
    
    print(batch_cmd)
    
    execute_script(batch_cmd)




# execute pipeline scripts in order
def folder_delete(file_path):
    
    isExists = os.path.exists(file_path)
    
    if isExists:
        
        shutil.rmtree(file_path)
    else:
        print("Path {} not exist!\n".format(file_path))
        

def create_folders(genotype_list):
    
    for item in genotype_list:
        
        folder_path = current_path + item + '/'
        
        mkdir(folder_path)


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    


def split_image(image_path, output_dir):
    
    rows = 5
    
    cols = 3
    
    should_square = False
    
    should_cleanup = False
    
    split_image(image_path, rows, cols, should_square, should_cleanup, [output_dir])


# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')

    return file_path, filename, basename


if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    ap.add_argument("-r", "--target_path", required = False, help = "path to target folders")
    #ap.add_argument("-tq", "--type_quaternion", required = True, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")
    args = vars(ap.parse_args())
    
    
   
    
    #parameter sets
    # path to individual folders
    current_path = args["path"]
    
    target_path = args["target_path"]
    
    subfolders = fast_scandir(current_path)
    
    #print("Processing folder in path '{}' ...\n".format(subfolders))

    
    #loop execute
    for subfolder_id, subfolder_path in enumerate(subfolders):
        
        '''
        ##################################################3
        folder_name = os.path.basename(subfolder_path)
        
        subfolder_path = os.path.join(subfolder_path, '')
        
        folder_name_short = folder_name.replace("MIArabidopsisRachaelAlex_Tray ", "")
        
        folder_name_short = folder_name_short.replace(".", "_")
        
        target_file = target_path + 'Tray_' + folder_name_short
        
        target_file = os.path.join(target_file, '')
        
        print("Moving file '{}' to {}\n".format(subfolder_path, target_file))
        
        shutil.move(subfolder_path, target_file) 
        
        #######################################################
        folder_name = os.path.basename(subfolder_path)
        
        source_file = subfolder_path + '/NIR/' + '0_0_0.png'

        target_file = target_path + folder_name + '.png'
        
        print("Moving file '{}' to '{}'\n".format(source_file, target_file))
        
        file_move(source_file, target_file)
        '''
        
        
        
        #folder_name = os.path.basename(subfolder_path)
        
        source_file_path = os.path.join(subfolder_path, '')

        # python3 trait_extract_AR_24.py -p ~/example/plant_test/test/Tray_1_2024-05-15_11-02-53_803_1300/ -ft png -min 5000 -md 30
        
        batch_cmd = "python3 trait_extract_AR_24.py -p " + source_file_path + " -ft png -min 5000 -md 45" 
        
        print(batch_cmd)
        
        execute_script(batch_cmd)
        
    
    '''
    #accquire image file list
    filetype = '*.png' 
    image_file_path = current_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    #print(imgList)
    
    for image_id, image in enumerate(imgList):
        
        (file_path, filename, basename) = get_file_info(image)
        
        #print(" file_path={} filename={} basename={}\n".format(file_path, filename, basename))
        
        mkpath = os.path.dirname(target_path) + '/' + basename + '/'
        mkdir(mkpath)
        
        save_path = os.path.join(mkpath, '')
        
        #split_image(image, save_path)
        
        batch_cmd = "split-image " + image + " 5 3 --output-dir " + save_path
    
        print(batch_cmd)
        
        execute_script(batch_cmd)
    '''
    

