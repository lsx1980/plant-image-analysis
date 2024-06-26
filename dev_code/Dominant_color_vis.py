'''
Name: Dominant_color_vis.py

Version: 1.0

Summary: Extract plant shoot traits (larea, solidity, max_width, max_height, avg_curv, color_cluster) by paralell processing 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2023-02-29

USAGE:

   Dominant_color_vis.py -p ~/example/AR_data/test/ -ft jpg 

'''



import cv2

from sklearn.cluster import KMeans
import glob
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import argparse

from rembg import remove



class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        img = cv2.imread(self.IMAGE)
        
        # PhotoRoom Remove Background API
        img = remove(img).copy()
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def plotClusters(self):
        #plotting 
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(projection='3d')
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        plt.show()
        
    def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure(figsize=(12,8))
        plt.axis("off")
        plt.imshow(chart)
        plt.show()
        
        
        

def plot_rgb(image_file, result_path):
    
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    
    base_name = os.path.splitext(os.path.basename(filename))[0]

    
    img = cv2.imread(image_file)
    
    # PhotoRoom Remove Background API
    #img = remove(img).copy()
    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    '''
    #get rgb values from image to 1D array
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()
    
    #plotting 
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(r, g, b)
    plt.show()
    '''
    
    # Creating figure
    fig = plt.figure(figsize = (16, 9))
    
    ax = plt.axes(projection ="3d")

    x = []
    y = []
    z = []
    c = []

    for row in range(0, img.shape[1]):
        
        for col in range(0, img.shape[0]):
            
            pix = img[col,row]
            
            newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)
                
            if(not newCol in c):
                x.append(pix[0]) 
                y.append(pix[1])
                z.append(pix[2])
                c.append(newCol)


    ax.scatter(x,y,z, c = c)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")

    # show plot
    result_file = result_path + base_name + '_color_vis.jpg'
    
    plt.savefig(result_file,  bbox_inches='tight')





if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", dest = "path", type = str, required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default='jpg,png', help = "Image filetype")
    ap.add_argument('-n', '--num_clusters', dest = "num_clusters", type = int, required = False, default = 2,  help = 'Number of clusters for K-means clustering (default 2, min 2).')
    args = vars(ap.parse_args())

    file_path = args["path"]
    
    ext = args['filetype'].split(',') if 'filetype' in args else []
    
    patterns = [os.path.join(file_path, f"*.{p}") for p in ext]
    
    files = [f for fs in [glob.glob(pattern) for pattern in patterns] for f in fs]
    
    
    
    num_clusters = args['num_clusters'] 



    #accquire image file list
    imgList = sorted(files)
    
   
    
    #loop execute
    for image_id, image in enumerate(imgList):
        
        plot_rgb(image, file_path)
    
    '''
    dc = DominantColors(file_dir, num_clusters) 
    
    colors = dc.dominantColors()
    
    dc.plotClusters()
    
    
    print(colors)
    '''

