#!/usr/bin/python

# import the necessary packages
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
import matplotlib.image as mpimg
import pylab as P

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Current directory for image files.")
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-m", "--mask", required = False, help = "Path to the mask image", default="None")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())

# setting path for results storage 
current_path = args["path"]
filename = args["image"]
image_path = current_path + filename

# construct result folder
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?
    # True
    # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        print path+'folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        print path+'path exists!'
        return False

# save folder construction
mkpath = current_path + str(filename[0:-4])

#if (args["mask"] == "None"):
	# call the function
mkdir(mkpath)
		
save_path = mkpath + '/'
print "results_folder: " + save_path



# load the image and convert it from BGR to RGB so that we can display it with matplotlib
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#obtain the dimension parameters of the image
(h, w) = image.shape[:2]


"""
# show our image for debugging
plt.figure(1)
plt.axis("off")
plt.imshow(image)
"""
# determine whether a mask image was inputed 
if (args["mask"] == "None"):
	# reshape the image to be a list of pixels
	pixels = image.reshape((image.shape[0] * image.shape[1], 3))

else:
	# set path
	mask_path = current_path + args["mask"]
	
	# load mask image as grayscale
	im_gray = cv2.imread(mask_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	#im_gray = cv2.resize(im_gray,(0,0),fx=0.1,fy=0.1)
	
	#extract the binary mask
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	mask =im_bw
	
	#masked_image_ori = cv2.bitwise_and(image,image,mask = im_bw)
	#masked_image = cv2.resize(masked_image_ori,(0,0),fx=0.1,fy=0.1)
	
	#apply the mask to get the segmentation of plant
	masked_image = cv2.bitwise_and(image,image,mask = im_bw)
	
	#set the black region into white for display
	#masked_image[np.where((masked_image == [0,0,0]).all(axis = 2))] = [71,89,40]
	#masked_image[np.where((masked_image == [0,0,0]).all(axis = 2))] = [255,255,255]

	#masked_image_path = save_path + 'masked.png'  
	#cv2.imwrite(masked_image_path,masked_image)
	
	# reshape the image to be a list of pixels
	pixels = masked_image.reshape((masked_image.shape[0] * masked_image.shape[1], 3))

############################################################
#Clustering process
###############################################################
# cluster the pixel intensities
clt = MiniBatchKMeans(n_clusters = args["clusters"])
#clt = KMeans(n_clusters = args["clusters"])
clt.fit(pixels)

#assign labels to each cluster 
labels = clt.fit_predict(pixels)

#obtain the quantized clusters using each label
quant = clt.cluster_centers_.astype("uint8")[labels]

# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image_rec = pixels.reshape((h, w, 3))


# build a histogram of clusters and then create a figure representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)


# remove the background color cluster
if (args["mask"] == "None"):
	clt.cluster_centers_ = clt.cluster_centers_[1: len(clt.cluster_centers_)]
else:
	clt.cluster_centers_ = clt.cluster_centers_[1: len(clt.cluster_centers_)]

#build a histogram of clusters using center lables
numLabels = utils.plot_centroid_histogram(clt)

#create a figure representing the distribution of each color
bar = utils.plot_colors(hist, clt.cluster_centers_)

#import necessay packages 
import os

#set the path
source = '/home/suxingliu/plantcv/' + 'Color_Distribution.png'
destination = save_path + (str(filename[0:-4]) + '_' +'Color_Distribution.png')
os.rename(source, destination)


# show our color bart and save bar figure
fig = plt.figure(3)
plt.title("Color Distributation Histogram")
plt.imshow(bar)
plt.xlabel("Percentage")
plt.ylabel("Color category")
frame = plt.gca()
#frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

#save bar image
name_of_file = (str(filename[0:-4]) + '_' +'bar.png')
complete_path = save_path + name_of_file       
plt.savefig(complete_path)

#plt.show()
plt.close(fig)

fig = plt.figure(5)
#clustered = np.hstack([image, quant])
clustered = quant
plt.imshow(clustered)
#plt.show()


#save clustered image
filename = args["image"]
fig_name = (str(filename[0:-4]) + '_' +'cluster_out.png')
fig_path_save = save_path + fig_name
mpimg.imsave(fig_path_save, clustered)
plt.close(fig)

#save mask image
if (args["mask"] == "None"):
	
	#read mask image as gray scale
	im_gray = cv2.imread(fig_path_save, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	#fill samll holes and area along edge of image
	from skimage.segmentation import clear_border
	
	# remove artifacts connected to image border
	cleared = im_bw.copy()
	im_bw_cleared = clear_border(cleared)
			
	#from skimage import morphology
	#im_bw_cleared = morphology.remove_small_objects(im_bw, 20000, connectivity=2)
	
	#remove small holes and objects 
	from scipy import ndimage as ndi
	label_objects, num_labels = ndi.label(im_bw_cleared)
	#print num_labels
	sizes = np.bincount(label_objects.ravel()) 
	mask_sizes = sizes > 500
	mask_sizes[0] = 0
	img_cleaned = mask_sizes[label_objects]
	
	#change output image type
	from skimage import img_as_ubyte
	img_cleaned = img_as_ubyte(img_cleaned)
	
	#save output mask image
	fig_name = (str(filename[0:-4]) + '_' +'mask.png')
	fig_path_mask = current_path + fig_name
	cv2.imwrite(fig_path_mask, img_cleaned)
	
	fig_path_mask = save_path + fig_name
	cv2.imwrite(fig_path_mask, img_cleaned)
	

