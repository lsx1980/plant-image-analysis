'''
Name: DominantColors.py

Version: 1.0

Summary: Extract dominant colors of an image
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2023-02-29

USAGE:

    time python3 DominantColors.py -p ~/example/Tara_data/test/ -ft jpg 

'''

import argparse
import os
import glob

import cv2
from sklearn.cluster import KMeans


class DominantColors:
    
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters):
        self.CLUSTERS = clusters
        self.IMAGE = image


    def dominantColors(self):
        # read image
        img_src = cv2.imread(self.IMAGE)

        # calculate the 50 percent of original dimensions
        width = int(img_src.shape[1])
        height = int(img_src.shape[0])

        # dsize
        dsize = (width, height)

        # convert to rgb from bgr
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS, random_state=0, n_init="auto")
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.centroid = kmeans.cluster_centers_

        # save labels
        self.label = kmeans.labels_

        # returning after converting to integer from float
        return self.centroid.astype(int), self.label
     
     
     
    def cluster_ratio(self):
     
        labels=list(self.label)
        
        percent=[]
        for i in range(len(self.centroid)):
            j = labels.count(i)
            j = j/(len(labels))
            percent.append(j)
        
        print(percent)
        return percent
        
        
        
    def optimal_n(self):
        
        #Elbow Method
        md=[]
        for i in range(1,21):
            kmeans=KMeans(n_clusters=i, random_state=0, n_init="auto")
            kmeans.fit(self.IMAGE)
            o=kmeans.inertia_
            md.append(o)
        print(md)



        
        
        
        
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    ap.add_argument('-n', "--n_cluster", type = int, required = False, default = 5,  help = 'Number of clusters for K-means clustering (default 2, min 2).')
    args = vars(ap.parse_args())
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args["filetype"]
    n_cluster = args["n_cluster"]
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    
    
    
    #loop execute
    for image_id, image in enumerate(imgList):
        
        dc = DominantColors(image, n_cluster) 
        
        colors = dc.dominantColors()
        
        print(colors)
        
        print(dc.cluster_ratio())
        
        #print(dc.optimal_n())
