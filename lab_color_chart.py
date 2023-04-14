'''
Name: trait_extract_parallel.py

Version: 1.0

Summary: Extract maize_tassel traits 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2022-09-29

USAGE:

    time python3 lab_color_chart.py -p ~/example/Tara_data/test/ -ft jpg -s YCC -c 2 -min 10000

'''


import os
import glob
import cv2
import numpy as np
import argparse
from pathlib import Path 


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import clear_border


import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

import pandas as pd
import plotly.express as px

#from mayavi import mlab
#from tvtk.api import tvtk







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
        print (path+' path exists!')
        return False


def image_BRG2LAB(image, base_name, file_extension):
    
    '''
    # extarct path and name of the image file
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    
    # extract the base name 
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # get the image file name
    image_file_name = Path(image_file).name
    
    print("Converting image {0} from RGB to LAB color space\n".format(str(image_file_name)))
    
    
    # load the input image 
    image = cv2.imread(image_file)
    '''
    # change to RGB space
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #plt.imshow(image_RGB)
    #plt.show()
    
    # get pixel color
    pixel_colors = image_RGB.reshape((np.shape(image_RGB)[0]*np.shape(image_RGB)[1], 3))
    
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    
    norm.autoscale(pixel_colors)
    
    pixel_colors = norm(pixel_colors).tolist()
    
    #pixel_colors_array = np.asarray(pixel_colors)
    
    #pixel_colors = pixel_colors.ravel()
    
    # change to lab space
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB )
    
   
    (L_chanel, A_chanel, B_chanel) = cv2.split(image_LAB)
    

    ######################################################################
   
    
    fig = plt.figure(figsize=(8.0, 6.0))
    
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    axis.scatter(L_chanel.flatten(), A_chanel.flatten(), B_chanel.flatten(), facecolors = pixel_colors, marker = ".")
    axis.set_xlabel("L:ightness")
    axis.set_ylabel("A:red/green coordinate")
    axis.set_zlabel("B:yellow/blue coordinate")
    
    # save segmentation result
    result_file = (save_path + base_name + '_lab' + file_extension)
    
    plt.savefig(result_file, bbox_inches = 'tight', dpi = 1000)
    
    ######################################################################
    #df_lab = pd.DataFrame({'height': height, 'weight': weight})
    '''
    mlab.figure("Structure_graph", size = (800, 800), bgcolor = (0, 0, 0))
    
    mlab.clf()
        
    x, y, z = L_chanel.flatten(), A_chanel.flatten(), B_chanel.flatten()

    pts = mlab.points3d(x,y,z, mode = 'point')

    sc = tvtk.UnsignedCharArray()

    sc.from_array(pixel_colors)

    pts.mlab_source.dataset.point_data.scalars = sc

    pts.mlab_source.dataset.modified()
    
    mlab.show()
    '''

    '''
    #fig = px.scatter_3d(L_chanel.flatten(), A_chanel.flatten(), B_chanel.flatten(), color= pixel_colors_array.flatten())
    fig = px.scatter_3d(L_chanel.flatten(), A_chanel.flatten(), B_chanel.flatten())
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    fig.show()
    '''

def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters):
    

    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)
            
    (width, height, n_channel) = image.shape
    
    #print("image shape: \n")
    #print(width, height, n_channel)
    
 
    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    

    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    
    #define number of cluster
    numClusters = max(2, args_num_clusters)
    
    # clustering method
    kmeans = KMeans(n_clusters = numClusters, n_init = 40, max_iter = 500).fit(reshaped)
    
    # get lables 
    pred_label = kmeans.labels_
    
    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)],key = lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i
    
    ret, thresh = cv2.threshold(kmeansImage,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #thresh_cleaned = (thresh)
    
    
    if np.count_nonzero(thresh) > 0:
        
        thresh_cleaned = clear_border(thresh)
    else:
        thresh_cleaned = thresh
    
     
    (nb_components, output, stats, centroids) = cv2.connectedComponentsWithStats(thresh_cleaned, connectivity = 8)
    
    
    # stats[0], centroids[0] are for the background label. ignore
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    area = stats[1:, cv2.CC_STAT_AREA]
    
    if sum(area) > (width*height*0.25):
        
        thresh_cleaned = ~thresh_cleaned

        (nb_components, output, stats, centroids) = cv2.connectedComponentsWithStats(thresh_cleaned, connectivity = 8)

    
    area = stats[1:, cv2.CC_STAT_AREA]
    
    Coord_left = stats[1:, cv2.CC_STAT_LEFT]
    
    Coord_top = stats[1:, cv2.CC_STAT_TOP]
    
    Coord_width = stats[1:, cv2.CC_STAT_WIDTH]
    
    Coord_height = stats[1:, cv2.CC_STAT_HEIGHT]
    
    Coord_centroids = centroids
    
    #print("Coord_centroids {}\n".format(centroids[1][1]))
    
    #print("[width, height] {} {}\n".format(width, height))
    
    nb_components = nb_components - 1
    
    
    
    max_size = width*height*0.1
    
    img_thresh = np.zeros([width, height], dtype=np.uint8)
    
    #for every component in the image, keep it only if it's above min_size
    for i in range(0, nb_components):
        
        '''
        #print("{} nb_components found".format(i))
        
        if (sizes[i] >= min_size) and (Coord_left[i] > 1) and (Coord_top[i] > 1) and (Coord_width[i] - Coord_left[i] > 0) and (Coord_height[i] - Coord_top[i] > 0) and (centroids[i][0] - width*0.5 < 10) and ((centroids[i][1] - height*0.5 < 10)) and ((sizes[i] <= max_size)):
            img_thresh[output == i + 1] = 255
            
            print("Foreground center found ")
            
        elif ((Coord_width[i] - Coord_left[i])*0.5 - width < 15) and (centroids[i][0] - width*0.5 < 15) and (centroids[i][1] - height*0.5 < 15) and ((sizes[i] <= max_size)):
            imax = max(enumerate(sizes), key=(lambda x: x[1]))[0] + 1    
            img_thresh[output == imax] = 255
            print("Foreground max found ")
        '''
        
        if (area[i] >= min_size):
        
            img_thresh[output == i + 1] = 255
        
    #from skimage import img_as_ubyte
    
    #img_thresh = img_as_ubyte(img_thresh)
    
    #print("img_thresh.dtype")
    #print(img_thresh.dtype)
    
    
    size_kernel = 40
    
    #if mask contains mutiple non-conected parts, combine them into one. 
    contours, hier = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 1:
        
        print("mask contains mutiple non-conected parts, combine them into one\n")
        
        kernel = np.ones((size_kernel,size_kernel), np.uint8)

        dilation = cv2.dilate(img_thresh.copy(), kernel, iterations = 1)
        
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        
        img_thresh = closing
        
    
    
    
    #return img_thresh
    return img_thresh



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
    



# compute all the traits
def extract_traits(image_file):


    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    
    base_name = os.path.splitext(os.path.basename(filename))[0]

    image_file_name = Path(image_file).name
   
    
    print("Exacting traits for image : {0}\n".format(str(image_file_name)))
     
    ##############################
    image = cv2.imread(image_file)
    
    # make backup image
    orig = image.copy()
    
    # get the dimension of the image
    img_height, img_width, img_channels = orig.shape

    #source_image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    

    ##########################################################################
    #Plant object detection
    x = int(0)
    y = int(0)
    w = int(img_width*0.4)
    h = int(img_height*1.0)

    roi_image = region_extracted(orig, x, y, w, h)
    
  
    
    
    #roi_image = orig.copy()
    
    orig = roi_image.copy()
    
    #orig = sticker_crop_img.copy()
    
    #color clustering based plant object segmentation
    thresh = color_cluster_seg(orig, args_colorspace, args_channels, args_num_clusters)
    
    
    #define result path for labeled images
    #result_file = (save_path + base_name + '_mask' + file_extension)
    #cv2.imwrite(result_file, thresh)
    
    #color clustering based plant object segmentation
    
    #apply the mask to get the segmentation of plant
    masked_image = cv2.bitwise_and(orig, orig, mask = thresh)

    #define result path for labeled images
    result_file = (save_path + base_name + '_masked' + file_extension)
    cv2.imwrite(result_file, masked_image)
    
    

    thresh_2 = color_cluster_seg(masked_image, 'lab', '2', args_num_clusters)
    
    #define result path for labeled images
    #result_file = (save_path + base_name + '_mask_2' + file_extension)
    #cv2.imwrite(result_file, thresh_2)
    
    
    #apply the mask to get the segmentation of plant
    masked_image_2 = cv2.bitwise_and(orig, orig, mask = thresh_2)

    #define result path for labeled images
    result_file = (save_path + base_name + '_masked_2' + file_extension)
    cv2.imwrite(result_file, masked_image_2)
    
    #image_BRG2LAB(masked_image_2, base_name, file_extension)
    
    
    
    



if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    ap.add_argument('-s', '--color-space', type = str, required = False, default ='lab', help='Color space to use: BGR, HSV, Lab, YCrCb (YCC)')
    ap.add_argument('-c', '--channels', type = str, required = False, default='1', help='Channel indices to use for clustering, where 0 is the first channel,' 
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" ' 
                                                                       + 'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type = int, required = False, default = 2,  help = 'Number of clusters for K-means clustering (default 2, min 2).')
    ap.add_argument('-min', '--min_size', type = int, required = False, default = 100,  help = 'min size of object to be segmented.')
    args = vars(ap.parse_args())
    
     # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    
    args_colorspace = args['color_space']
    args_channels = args['channels']
    args_num_clusters = args['num_clusters']
    
    min_size = args['min_size']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    # save folder construction
    mkpath = os.path.dirname(file_path) +'/lab_color_space'
    mkdir(mkpath)
    save_path = mkpath + '/'
    print ("results_folder: " + save_path)
    
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    n_images = len(imgList)
    
    
    #loop execute
    for image_id, image_file in enumerate(imgList):
        
        extract_traits(image_file)
        

    
    
    
    '''
    # get cpu number for parallel processing
    agents = psutil.cpu_count() - 2 
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(image_BRG2LAB, imgList)
        pool.terminate()
    '''
