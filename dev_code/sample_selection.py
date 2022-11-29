'''
Name: down_sample.py

Version: 1.0

Summary: Select sample images among image list
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2019-09-29

USAGE:

python3 sample_selection.py -p ~/example/plant_test/seeds/PhenoTools_Data_2022/ -ft png -key 'ANTQA'


'''

# import the necessary packages
import os
import glob
import argparse
import shutil
import natsort 


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
        


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required = True,    help="Image filetype")
    ap.add_argument("-key", "--keywords", required = True, type = str, help="key words in file name")
    args = vars(ap.parse_args())
    

    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    kw = str(args['keywords'])
    
    # make the folder to store the results
    # save folder construction
    mkpath = os.path.dirname(file_path) +'/images'
    mkdir(mkpath)
    save_path = mkpath + '/'
    print("results_folder: {0}\n".format(str(save_path)))  
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    imgList = natsort.natsorted(imgList,reverse = False)
    
    #print(imgList)
    #sub = 'DECDR'
    
    selected_image_list = (s for s in imgList if kw.lower() in s.lower())
    
    #print ("\n".join(s for s in imgList if sub.lower() in s.lower()))
    

    
     
    #loop execute
    for (i, image_file) in enumerate(selected_image_list):

        filename, file_extension = os.path.splitext(image_file)
    
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        #print(base_name)
        
        dst = (save_path + base_name + '.' + ext)

        shutil.move(image_file, dst)
        
       
            
        
        
