'''
Name: color_segmentation.py

Version: 1.0

Summary:  Extract plant traits (leaf area, width, height, ) by paralell processing 
    
Author: suxing liu

Author-email: suxingliu@gmail.com


USAGE:

    python3 Dominant_color_clustering.py -p ~/example/plant_test/mi_test/test/ -ft png -c 5


'''

#!/usr/bin/python

# import the necessary packages

import os
import glob
import argparse

from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation
import colorsys

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly



    
    
def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r=int(rgb[0])
    g=int(rgb[1])
    b=int(rgb[2])
    return '#%02x%02x%02x' % (r,g,b)
    
    
    
def mkdir(path):
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
        print (path+'folder constructed!')
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        print (path+'path exists!')
        return False
        

def color_cluster_visualization(image_file):
    
    im = Image.open(image_file)

    w, h = im.size

    colors = im.getcolors(w*h)

    '''
    for idx, c in enumerate(colors):
        plt.bar(idx, c[0], color=hexencode(c[1]), lw=1, ec='k')
    plt.ylabel('Pixels')
    plt.xlabel('Color')
    plt.savefig(save_path + 'color-hist.png', bbox_inches='tight')
    '''


    df = pd.DataFrame(
    data={
        'pixels': [colors[i][0] for i in range(len(colors))],
        'R': [colors[i][1][0] for i in range(len(colors))],
        'G': [colors[i][1][1] for i in range(len(colors))],
        'B': [colors[i][1][2] for i in range(len(colors))],
        #'alpha': [colors[i][1][3] for i in range(len(colors))],
        'hex': [hexencode(colors[i][1]) for i in range(len(colors))]
    })
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.scatter(x=df.R, y=df.B, s=30, c=df.hex, alpha=.6, edgecolor='k', lw=0.3)
    plt.axis([0, 255, 0, 255])
    plt.xlabel('Red', fontsize=14)
    plt.ylabel('Blue', fontsize=14)
    plt.subplot(122)
    plt.scatter(x=df.G, y=df.B, s=40, c=df.hex, alpha=.6, edgecolor='k', lw=0.3)
    plt.axis([0, 255, 0, 255])
    plt.xlabel('Green', fontsize=14)
    plt.ylabel('Blue', fontsize=14)
    plt.savefig(save_path + 'rgb-proj.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = df.R
    y = df.G
    z = df.B
    c = df.hex
    s = 30

    ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    plt.savefig(save_path + 'rgb-scatter.png', bbox_inches='tight')


    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = df.R
    y = df.G
    z = df.B
    c = df.hex
    s = df.pixels * 15

    ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    plt.savefig(save_path + 'rgb-scatter2.png', bbox_inches='tight')

    #RGB k-means
    ########################################################################
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(df[['R', 'G', 'B']])
    df['kcenter'] = kmeans.labels_
    
    avg_col = np.zeros((kmeans.n_clusters, 3))
    
    for c in range(kmeans.n_clusters):
        temp_df = df[df.kcenter == c]
        avg_col[c, 0] = np.average(temp_df.R, weights=temp_df.pixels)
        avg_col[c, 1] = np.average(temp_df.B, weights=temp_df.pixels)
        avg_col[c, 2] = np.average(temp_df.G, weights=temp_df.pixels)
    
    hsv_matrix = np.zeros((len(df), 3))

    for i in range(len(df)):
        hsv_matrix[i] = colorsys.rgb_to_hsv(r=df.R[i]/255, g=df.G[i]/255, b=df.B[i]/255)
        
    df['h'] = hsv_matrix[:, 0]
    df['s'] = hsv_matrix[:, 1]
    df['v'] = hsv_matrix[:, 2]

    avg_col2 = np.zeros((kmeans.n_clusters, 3))
    for c in range(kmeans.n_clusters):
        temp_df = df[df.kcenter == c]
        avg_col2[c, 0], avg_col2[c, 1], avg_col2[c, 2] = colorsys.hsv_to_rgb(h=np.average(temp_df.h, weights=temp_df.pixels),
                                                                             s=np.average(temp_df.s, weights=temp_df.pixels),
                                                                             v=np.average(temp_df.v, weights=temp_df.pixels))
    avg_col2 *= 255
    
    
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(131, projection='3d')

    x = kmeans.cluster_centers_[:, 0]
    y = kmeans.cluster_centers_[:, 1]
    z = kmeans.cluster_centers_[:, 2]
    c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
    s = 300

    ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    ax.set_title('RGB K-means', fontsize=16)
    plt.savefig(save_path + 'RGB_K-means.png', bbox_inches='tight')
    


    ax = fig.add_subplot(132, projection='3d')
    x = avg_col[:, 0]
    y = avg_col[:, 1]
    z = avg_col[:, 2]
    c = [hexencode(r) for r in avg_col]
    s = 300

    ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    ax.set_title('Weighted average RGB', fontsize=16)
    plt.savefig(save_path + 'Weighted_average_RGB.png', bbox_inches='tight')


    ax = fig.add_subplot(133, projection='3d')
    x = avg_col2[:, 0]
    y = avg_col2[:, 1]
    z = avg_col2[:, 2]
    c = [hexencode(r) for r in avg_col2]
    s = 300

    ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    ax.set_title('Weighted average HSV', fontsize=16)
    plt.savefig(save_path + 'Weighted_average_HSV.png', bbox_inches='tight')
    
    
    #HSV k-means
    #############################################################################################3
    kmeansHSV = KMeans(n_clusters=4, random_state=0, n_init=10).fit(df[['h', 's', 'v']])
    
    dfHSV = df.copy()
    dfHSV['kcenter'] = kmeansHSV.labels_

    HSVcenters = np.zeros((kmeansHSV.n_clusters, 3))
    for i in range(kmeansHSV.n_clusters):
        HSVcenters[i, :] = colorsys.hsv_to_rgb(h=kmeansHSV.cluster_centers_[i, 0],
                                               s=kmeansHSV.cluster_centers_[i, 1],
                                               v=kmeansHSV.cluster_centers_[i, 2])
    HSVcenters *= 255

    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = kmeans.cluster_centers_[:, 0]
    y = kmeans.cluster_centers_[:, 1]
    z = kmeans.cluster_centers_[:, 2]
    c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
    s = 400

    ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)


    x = HSVcenters[:, 0]
    y = HSVcenters[:, 1]
    z = HSVcenters[:, 2]
    c = [hexencode(HSVcenters[i,:]) for i in range(kmeansHSV.n_clusters)]

    ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1, marker='s')

    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    ax.set_title('Comparison of K-means', fontsize=16)
    ax.legend(['RGB k-means', 'HSV k-means'], scatterpoints=1, frameon=False, fontsize=13)
    plt.savefig(save_path + 'RGB_HSV_cluster.png', bbox_inches='tight')

    
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = df.h
    y = df.s
    z = df.v
    c = df.hex
    s = 30

    ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)

    # ax.set_xlim3d(0, 255)
    # ax.set_ylim3d(0, 255)
    # ax.set_zlim3d(0, 255)

    ax.set_xlabel('H', fontsize=14)
    ax.set_ylabel('S', fontsize=14)
    ax.set_zlabel('V', fontsize=14)
    
    x = df.h
    y = df.s
    z = df.v
    c = df.hex
    s = 30

    plt.scatter(x, y, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)
    plt.savefig(save_path + 'Circular_HSV.png', bbox_inches='tight')
    
    #print(df[['h', 's', 'v']].describe())
    
    
    circ_y = df.s*np.sin(df.h*2*np.pi)
    circ_x = df.s*np.cos(df.h*2*np.pi)
    
    plt.figure(figsize=(16,6))
    ax = plt.subplot(121)
    plt.scatter(circ_x, circ_y, s=40, alpha = .25, c=df.hex, lw=0)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)

    ax = plt.subplot(122)
    plt.scatter(df.v*circ_x, df.v*circ_y, s=40, alpha = .25, c=df.hex, lw=0)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)

    plt.savefig(save_path + 'hsv-proj.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = df.v*df.s*np.sin(df.h*2*np.pi)
    y = df.v*df.s*np.cos(df.h*2*np.pi)
    z = df.v
    c = df.hex
    s = 30

    ax.scatter(x, y, z, c=c, s=s, alpha=.2, edgecolor='k', lw=0)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 1)

    # ax.set_xlabel('H', fontsize=14)
    # ax.set_ylabel('S', fontsize=14)
    ax.set_zlabel('V', fontsize=14)

    ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
    ax.tick_params(
    axis='y',
    which='both',
    bottom=False,
    top=False,
    right=False,
    left=False,
    labelbottom=False,
    labelright=False,
    labelleft=False)

    plt.savefig(save_path + 'hsv-scatter.png', bbox_inches='tight')

    cut = df[df.v > 0.40]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = cut.v*cut.s*np.sin(cut.h*2*np.pi)
    y = cut.v*cut.s*np.cos(cut.h*2*np.pi)
    z = cut.v
    c = cut.hex
    s = 30

    ax.scatter(x, y, z, c=c, s=s, alpha=.2, edgecolor='k', lw=0)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 1)

    # ax.set_xlabel('H', fontsize=14)
    # ax.set_ylabel('S', fontsize=14)
    ax.set_zlabel('V', fontsize=14)
    plt.savefig(save_path + 'hsv-scatter_cut.png', bbox_inches='tight')
    
    
    
    ################################################
    
    weighted_cluster_centers = np.zeros((kmeans.n_clusters, 3))
    for c in range(kmeans.n_clusters):
        weighted_cluster_centers[c] = np.average(df[df['kcenter'] == c][['R', 'G', 'B']], weights=df[df['kcenter'] == c]['pixels'], axis=0)

    x = df.R
    y = df.G
    z = df.B
    c = df.hex

    trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=6,
        color=c,
        opacity=0.25),
    name='Individual colors')

    x = kmeans.cluster_centers_[:, 0]
    y = kmeans.cluster_centers_[:, 1]
    z = kmeans.cluster_centers_[:, 2]
    c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]

    trace2 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=16,
        color=c,
        opacity=1),
    name='k-means center')

    x = weighted_cluster_centers[:, 0]
    y = weighted_cluster_centers[:, 1]
    z = weighted_cluster_centers[:, 2]
    c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
    trace3 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=16,
        color="#aa00aa",
        opacity=1),
    name='weighted center')

    data = [trace1, trace2, trace3]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    #_ = iplot(fig)
    plotly.offline.plot(fig, auto_open = False, filename = save_path + 'ploty_RGB_cluster.html')
    
    
    
    
    
    
    weighted_cluster_centers = np.zeros((kmeansHSV.n_clusters, 3))
    for c in range(kmeansHSV.n_clusters):
        # weighted_cluster_centers[c] = dfHSV[dfHSV['kcenter'] == c].mean()[['h', 's', 'v']].values
        weighted_cluster_centers[c] = np.average(dfHSV[dfHSV['kcenter'] == c][['h', 's', 'v']], weights=dfHSV[dfHSV['kcenter'] == c]['pixels'], axis=0)

    weighted_cluster_center_colors = np.zeros((kmeansHSV.n_clusters, 3))
    for i in range(kmeansHSV.n_clusters):
        weighted_cluster_center_colors[i] = colorsys.hsv_to_rgb(
            h=weighted_cluster_centers[i, 0],
            s=weighted_cluster_centers[i, 1],
            v=weighted_cluster_centers[i, 2])
    weighted_cluster_center_colors *= 255

    x = dfHSV.v * dfHSV.s * np.sin(dfHSV.h * 2 * np.pi)
    y = dfHSV.v * dfHSV.s * np.cos(dfHSV.h * 2 * np.pi)
    z = dfHSV.v
    c = dfHSV.hex

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=c,
            opacity=0.25),
        name='Individual colors')

    x = kmeansHSV.cluster_centers_[:, 2] * kmeansHSV.cluster_centers_[:, 1] * np.sin(kmeansHSV.cluster_centers_[:, 0] * 2 * np.pi)
    y = kmeansHSV.cluster_centers_[:, 2] * kmeansHSV.cluster_centers_[:, 1] * np.cos(kmeansHSV.cluster_centers_[:, 0] * 2 * np.pi)
    z = kmeansHSV.cluster_centers_[:, 2]
    c = [hexencode(HSVcenters[i,:]) for i in range(kmeansHSV.n_clusters)]

    trace2 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=16,
            color=c,
            opacity=1),
        name='k-means center')

    x = weighted_cluster_centers[:, 2] * weighted_cluster_centers[:, 1] * np.sin(weighted_cluster_centers[:, 0] * 2 * np.pi)
    y = weighted_cluster_centers[:, 2] * weighted_cluster_centers[:, 1] * np.cos(weighted_cluster_centers[:, 0] * 2 * np.pi)
    z = weighted_cluster_centers[:, 2]
    c = [hexencode(weighted_cluster_center_colors[i]) for i in range(kmeansHSV.n_clusters)]

    trace3 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=16,
            color="#aa00aa",
            opacity=1),
        name='weighted center')

    data = [trace1, trace2, trace3]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, auto_open = False, filename = save_path + 'ploty_HSV_cluster.html')
    
    
    



if __name__ == '__main__':
        
        
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Current directory for image files.")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    #ap.add_argument("-m", "--mask", required = True, help = "Path to the mask image", default = "None")
    ap.add_argument("-c", "--clusters", required = True, type = int, help = "Number of clusters")
    args = vars(ap.parse_args())

    # setting path for results storage 
    file_path = args["path"]
    ext = args['filetype']
    n_clusters = args['clusters']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    # construct result folder
    mkpath = os.path.dirname(file_path) +'/results'
        
    mkdir(mkpath)
    
    global save_path
    save_path = mkpath + '/'
    print ("results_folder: " + save_path)
    
    


    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    for image_id, image_file in enumerate(imgList):
    
        color_cluster_visualization(image_file)
        
        

    


    




