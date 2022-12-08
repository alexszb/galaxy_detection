import pathlib
from traceback import format_exc
from PIL import Image, ImageDraw
import numpy as np
import os
import time
import tensorflow as tf
from keras import models

# global total_detected
total_detected = 0
project_dir = os.path.dirname(__file__)
cnn_model = tf.keras.models.load_model("%s/cnn/cnn_main.hdf5" % project_dir)
ring_model = tf.keras.models.load_model("%s/cnn_ring/cnn_ring.hdf5" % project_dir)

def make_tiles(ndarray, new_shape):
    pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        ndarray = ndarray.mean(-1*(i+1))
    return ndarray


def ring_classification(img_arrays):
    predictions = ring_model.predict(img_arrays)
    return predictions

def classification(image_arr, square_cords_list, fit_coords_list):
    detected_galaxies = []

    img_arrays = []
    fit_coords_filtered = []

    for i in range(len(square_cords_list)):
        square_cords = square_cords_list[i]
        if (square_cords[0] > 0 and square_cords[2] < 1984 and square_cords[1] > 0 and square_cords[3] < 1361):
            img_arrays.append(image_arr[square_cords[1]:square_cords[3],square_cords[0]:square_cords[2]])
            # img_arrays.append(image_arr[600:728,920:1048])
            fit_coords_filtered.append(fit_coords_list[i])

    img_array = np.asarray(img_arrays)
    predictions = cnn_model.predict(img_array)
    score = tf.nn.softmax(predictions)
    classes = np.argmax(score, axis=1)
    
    spiral_inds = np.where(classes == 2)[0]
    ring_predictions = ring_classification(img_array[spiral_inds])
    ring_inds = np.where(ring_predictions > 0.5)[0]
    ringed = np.zeros(classes.shape)
    ringed[ring_inds] = 1

    for i in range(len(classes.tolist())):
        if classes[i] == 1:
            detected_galaxies.append([fit_coords_filtered[i], "elliptical", 0])
        if classes[i] == 2:
            if ringed[i] == 1:
                detected_galaxies.append([fit_coords_filtered[i], "spiral", 1])
            else:
                detected_galaxies.append([fit_coords_filtered[i], "spiral", 0])
        if classes[i] == 0:
            detected_galaxies.append([fit_coords_filtered[i], "background", 0])

    return detected_galaxies    
    # im.show()        

def dbscan(filename):
    detected_galaxies = []
    solution_dir = pathlib.Path(project_dir).parent
    # im = Image.open("%s/faster_rcnn/data/test/%s.jpg" % (solution_dir, filename))
    im = Image.open(filename)
    draw = ImageDraw.ImageDraw(im)

    image_arr = np.array(im)

    intensity_arr = np.sum(image_arr, axis=2)

    intensity_arr = intensity_arr[0:(int(image_arr.shape[0]/16)*16),0:(int(image_arr.shape[1]/16)*16)]

    small_arr = make_tiles(intensity_arr, new_shape=(int(intensity_arr.shape[0]/8),int(intensity_arr.shape[1]/8)))

    cutoff = 50

    minPts = 8

    #changed from 2 to 3, testing
    eps = 2
    clusters = []
    already_seen = np.zeros((small_arr.shape[0],small_arr.shape[1]))

    for i in range(small_arr.shape[0]):
        for j in range(small_arr.shape[1]):
            if(already_seen[i,j] == 0):
                ymin = max(0, (i-eps))
                xmin = max(0, (j-eps))
                ymax = min(small_arr.shape[0]-2, (i+eps))
                xmax = min(small_arr.shape[1]-2, (j+eps))
                q1 = small_arr[ymin:ymax+1, xmin:xmax+1]
                q2 = np.asarray(np.where(q1 > cutoff ))
                if(q2.shape[1] >= minPts):
                    q2[0,:] += i
                    q2[1,:] += j
                    cluster = []
                    z_limit = q2.shape[1]
                    z = 0
                    while(z < z_limit):
                        if(q2[0][z] >= 0 and q2[1][z] >= 0 and q2[0][z] < small_arr.shape[0]-2 and q2[1][z] < small_arr.shape[1]-2 and already_seen[q2[0][z],q2[1][z]] == 0):
                            cluster.append(np.asarray((q2[0][z],q2[1][z])))
                            already_seen[q2[0][z],q2[1][z]] = 1
                            ymin_inner = max(0, (q2[0][z]-eps))
                            xmin_inner = max(0, (q2[1][z]-eps))
                            ymax_inner = min(small_arr.shape[0]-2, (q2[0][z]+eps))
                            xmax_inner = min(small_arr.shape[1]-2, (q2[1][z]+eps))
                            d1 = small_arr[ymin_inner:ymax_inner+1, xmin_inner:xmax_inner+1]
                            d2 = np.asarray(np.where(d1 > cutoff ))
                            if(d2.shape[1] >= minPts):
                                d2[0,:] += q2[0][z]-eps
                                d2[1,:] += q2[1][z]-eps
                                q2 = np.hstack((q2,d2))
                                z_limit += d2.shape[1]
                        z+=1
                    clusters.append(cluster)

    boxes = []
    for c in clusters:
        sh = np.asarray(c).shape
        if(sh[0] > 16):
            mins = np.min(c, axis=0)
            maxs = np.max(c, axis=0)
            width = maxs[1]-mins[1]
            height = maxs[0]-mins[0]
            if(width > 2 and height > 2):
                boxes.append(np.asarray((mins[0],mins[1],maxs[0],maxs[1])))
            
    # print(len(boxes))

    inds = np.where(small_arr >= cutoff)

    # for n in range(len(inds[0])):
    #     # print("%d,%d" % (inds[0][n]*16,inds[1][n]*16))
    #     yCtr = inds[0][n]*8+4
    #     xCtr = inds[1][n]*8+4
    #     draw.line((xCtr-4, yCtr-4, xCtr+4, yCtr-4), fill='red')
    #     draw.line((xCtr-4, yCtr+4, xCtr+4, yCtr+4), fill='red')
    #     draw.line((xCtr-4, yCtr-4, xCtr-4, yCtr+4), fill='red')
    #     draw.line((xCtr+4, yCtr-4, xCtr+4, yCtr+4), fill='red')

    square_coords_list = []
    fit_coords_list = []

    for n in range(len(boxes)):
        toplefty = boxes[n][0]*8
        topleftx = boxes[n][1]*8
        bottomrighty = boxes[n][2]*8
        bottomrightx = boxes[n][3]*8

        centerx = topleftx+int((bottomrightx-topleftx)/2)
        centery = toplefty+int((bottomrighty-toplefty)/2)
        width = bottomrightx-topleftx
        height = bottomrighty-toplefty
        side = max(width, height)
        fit_coords = (topleftx, toplefty, bottomrightx+8, bottomrighty+8)
        fit_coords_list.append(fit_coords)
        side = 128
        toplefty = centery-int(side/2)
        topleftx = centerx-int(side/2)
        bottomrighty = centery+int(side/2)
        bottomrightx = centerx+int(side/2)
        square_coords = (topleftx, toplefty, bottomrightx, bottomrighty)
        square_coords_list.append(square_coords)

        a = fit_coords
        # draw.line((a[0], a[1], a[0], a[3]),fill='yellow', width=3)
        # draw.line((a[2], a[1], a[2], a[3]),fill='yellow', width=3)
        # draw.line((a[0], a[1], a[2], a[1]),fill='yellow', width=3)
        # draw.line((a[0], a[3], a[2], a[3]),fill='yellow', width=3)
    detected_galaxies.extend(classification(image_arr, square_coords_list, fit_coords_list))
    # im.show()
    return detected_galaxies

def readBoxesFromFile(filename):
    gt_boxes = np.empty((0,6), int)
    with open(filename) as openfileobject:
        for line in openfileobject:
            data = line.split(',')
            if(data[5] == "ringed_spirals\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 1, 1]
                    gt_boxes = np.vstack((gt_boxes,gt_box))
            if(data[5] == "non_ringed_spirals\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 1, 0]
                    gt_boxes = np.vstack((gt_boxes,gt_box))
            if(data[5] == "ellipticals\n"):
                gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 2, 0]
                gt_boxes = np.vstack((gt_boxes,gt_box)) 
    return gt_boxes

def show_gt_boxes(filename):
    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    imageNumber = filename.split("/")[-1].split(".")[0]
    gt_boxes = readBoxesFromFile("%s/faster_rcnn/data/test/%s.txt" % (solution_dir, imageNumber))
    for box in range(len(gt_boxes)):
        a = gt_boxes[box]
        color = "white"
        if (a[4] == 1):
            if (a[5] == 1):
                color = "green"
            else:
                color = "blue"
        else:
            color = "red"
        draw.line((a[0], a[1], a[0], a[3]),fill=color)
        draw.line((a[2], a[1], a[2], a[3]),fill=color)
        draw.line((a[0], a[1], a[2], a[1]),fill=color)
        draw.line((a[0], a[3], a[2], a[3]),fill=color)
    im.save("%s/detector_gui/results/%s_original.jpg" % (solution_dir, imageNumber))

# dbscan("C:/Users/Alex/source/repos/ImageGenerator/maybefortest/30002.jpg")