from turtle import back
from PIL import Image, ImageDraw
import random
import os
import math
import numpy as np
import tensorflow as tf
import pathlib
from utils import arrayAdder
from utils.logcorrect import log_correction

def drawGalaxy(galaxy_type, solution_dir, count, train_val_test, im_height, im_width):
    s = 0
    while (s < 6):
        galaxies = os.listdir("%s/images/%s/clean/" % (solution_dir, galaxy_type))
        train_galaxies_size = int(len(galaxies)*0.7)
        val_test_galaxies_start= int(len(galaxies)*0.2)+train_galaxies_size
        if(train_val_test == 0):
            galaxy = Image.open("%s/images/%s/clean/%s" % (solution_dir, galaxy_type, galaxies[random.randrange(0, train_galaxies_size)]))
        if(train_val_test == 1):
            galaxy = Image.open("%s/images/%s/clean/%s" % (solution_dir, galaxy_type, galaxies[random.randrange(train_galaxies_size, val_test_galaxies_start)]))
        if(train_val_test == 2):
            galaxy = Image.open("%s/images/%s/clean/%s" % (solution_dir, galaxy_type, galaxies[random.randrange(val_test_galaxies_start, len(galaxies))]))

        corrected_galaxy = log_correction(np.asarray(galaxy))
        galaxy = Image.fromarray(corrected_galaxy)

        rot = random.randint(0,359)

        im_size = min(galaxy.width, galaxy.height)
        rsize = random.uniform(im_size/96,im_size/160)

        galaxyc = galaxy.resize((int(galaxy.height/rsize),int(galaxy.width/rsize)))
        galaxyc = galaxyc.rotate(-rot, expand=True)

        galaxyc_array = tf.keras.preprocessing.image.img_to_array(galaxyc)



        padding_top= random.randint(0, im_height-galaxyc.height)

        padding_left = random.randint(0, im_width-galaxyc.width)

        global imToFill_array
        imToFill_array=arrayAdder.fillWithGalaxy(imToFill_array,galaxyc_array, padding_top, padding_left, galaxyc.height, galaxyc.width)

        x1 = padding_left
        y1 = padding_top
        x2 = padding_left + galaxyc_array.shape[1]
        y2 = padding_top + galaxyc_array.shape[0]

        if(train_val_test == 0):
            with open("%s/faster_rcnn/data/train/%s.txt" % (solution_dir, count), 'a') as labels:
                labels.write("%s.jpg,%d,%d,%d,%d,%s\n" % (count,x1,y1,x2,y2,galaxy_type))
        if(train_val_test == 1):
            with open("%s/faster_rcnn/data/val/%s.txt" % (solution_dir, count), 'a') as labels:
                labels.write("%s.jpg,%d,%d,%d,%d,%s\n" % (count,x1,y1,x2,y2,galaxy_type))
        if(train_val_test == 2):
            with open("%s/faster_rcnn/data/test/%s.txt" % (solution_dir, count), 'a') as labels:
                labels.write("%s.jpg,%d,%d,%d,%d,%s\n" % (count,x1,y1,x2,y2,galaxy_type))
        s += 1     

def generateSkyImage(solution_dir, count, train_val_test):
    if not (os.path.exists("%s/faster_rcnn/data" % solution_dir)):
        os.makedirs("%s/faster_rcnn/data" % solution_dir)
    if not (os.path.exists("%s/faster_rcnn/data/train/" % solution_dir)):
        os.makedirs("%s/faster_rcnn/data/train" % solution_dir)
    if not (os.path.exists("%s/faster_rcnn/data/val/" % solution_dir)):
        os.makedirs("%s/faster_rcnn/data/val" % solution_dir)
    if not (os.path.exists("%s/faster_rcnn/data/test/" % solution_dir)):
        os.makedirs("%s/faster_rcnn/data/test" % solution_dir)


    backgrounds = os.listdir("%s/images/backgrounds/original/" % solution_dir)

    train_bg_size = int(len(backgrounds)*0.7)
    val_test_bg_start= int(len(backgrounds)*0.2)+train_bg_size
    if (train_val_test == 0):
        imToFill = Image.open("%s/images/backgrounds/original/%s" % (solution_dir, backgrounds[random.randrange(0, train_bg_size)]))
    if (train_val_test == 1):
        imToFill = Image.open("%s/images/backgrounds/original/%s" % (solution_dir, backgrounds[random.randrange(train_bg_size, val_test_bg_start)]))
    if (train_val_test == 2):
        imToFill = Image.open("%s/images/backgrounds/original/%s" % (solution_dir, backgrounds[random.randrange(val_test_bg_start, len(backgrounds))]))       
    
    if(random.randint(0,1) == 1):
        imToFill = imToFill.transpose(Image.FLIP_LEFT_RIGHT)
    if(random.randint(0,1) == 1):
        imToFill = imToFill.transpose(Image.FLIP_TOP_BOTTOM)

    global imToFill_array
    imToFill_array = tf.keras.preprocessing.image.img_to_array(imToFill)

    drawGalaxy("non_ringed_spirals", solution_dir, count, train_val_test, imToFill.height, imToFill.width)
    drawGalaxy("ellipticals", solution_dir, count, train_val_test, imToFill.height, imToFill.width)
    drawGalaxy("ellipticals", solution_dir, count, train_val_test, imToFill.height, imToFill.width)
    drawGalaxy("ringed_spirals", solution_dir, count, train_val_test, imToFill.height, imToFill.width)

    if(train_val_test == 0):
        tf.keras.utils.save_img("%s/faster_rcnn/data/train/%s.jpg" % (solution_dir, count), imToFill_array, data_format=None, file_format=None, scale=True)
    if(train_val_test == 1):
        tf.keras.utils.save_img("%s/faster_rcnn/data/val/%s.jpg" % (solution_dir, count), imToFill_array, data_format=None, file_format=None, scale=True)
    if(train_val_test == 2):
        tf.keras.utils.save_img("%s/faster_rcnn/data/test/%s.jpg" % (solution_dir, count), imToFill_array, data_format=None, file_format=None, scale=True)

    if(count % 100 == 0):
        print("Image generated... %s" % count)

def getBoundigboxCoordinates(im, im1, rot, resize, cx, cy):
    rad = math.radians(rot)

    rx = (im.width/2)/resize
    ry = (im.height/2)/resize

    topleftx = int(-rx*math.cos(rad)-(-ry*math.sin(rad))+(((im1.width-im.width)/2))+(rx*resize))
    toplefty = int(-rx*math.sin(rad)+(-ry*math.cos(rad))+(((im1.height-im.height)/2))+(ry*resize))

    toprightx = int(rx*math.cos(rad)-(-ry*math.sin(rad))+((im1.width-im.width)/2)+(rx*resize))
    toprighty = int(rx*math.sin(rad)+(-ry*math.cos(rad))+((im1.height-im.height)/2)+(ry*resize))

    bottomrightx = int(rx*math.cos(rad)-(ry*math.sin(rad))+((im1.width-im.width)/2)+(rx*resize))
    bottomrighty = int(rx*math.sin(rad)+(ry*math.cos(rad))+((im1.height-im.height)/2)+(ry*resize))

    bottomleftx = int(-rx*math.cos(rad)-(ry*math.sin(rad))+((im1.width-im.width)/2)+(rx*resize))
    bottomlefty = int(-rx*math.sin(rad)+(ry*math.cos(rad))+((im1.height-im.height)/2)+(ry*resize))

    topleftx += cx
    toplefty += cy
    toprightx += cx
    toprighty += cy
    bottomleftx += cx
    bottomlefty += cy
    bottomrightx += cx
    bottomrighty += cy

    return ((topleftx,toprightx,bottomleftx, bottomrightx),(toplefty, toprighty, bottomlefty, bottomrighty))

def readBoxesFromFile(filename):
    gt_boxes = np.empty((0,4), int)
    with open(filename) as openfileobject:
        for line in openfileobject:
            data = line.split(',')
            gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4])]
            gt_boxes = np.vstack((gt_boxes,gt_box))
    return gt_boxes

def showBoundingBoxes(im):
    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent
    gt_boxes = readBoxesFromFile("%s/images/generated/train/%s.txt" % (solution_dir, im))
    showanchors = Image.open("%s/images/generated/train/%s.jpg " % (solution_dir, im))
    draw = ImageDraw.Draw(showanchors)
    for t in range(len(gt_boxes)):  
                a = gt_boxes[t]
                draw.line((a[0], a[1], a[0], a[3]),fill='white')
                draw.line((a[2], a[1], a[2], a[3]),fill='white')
                draw.line((a[0], a[1], a[2], a[1]),fill='white')
                draw.line((a[0], a[3], a[2], a[3]),fill='white')
    showanchors.show()

# for m in range(1):
#     showBoundingBoxes(m)