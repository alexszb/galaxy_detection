from email.mime import image
from tkinter import *
from tkinter import filedialog as fd
from traceback import format_exc
from PIL import ImageDraw, Image, ImageTk
import tensorflow as tf
import numpy as np
import keras.backend as K
from utils.util import getAllAnchors, bbox_transform_inv, non_max_suppression_gpu, non_max_suppression_gpu2, loss_cls, smoothL1, RoI_Align
import gc
import os
import pathlib
import gzip
import shutil
import sys

project_dir = os.path.dirname(__file__)
p = pathlib.Path(project_dir)
solution_dir = p.parent

rpn_model = tf.keras.models.load_model('%s/faster_rcnn/rpn/rpn_model.hdf5' % solution_dir, custom_objects={'loss_cls': loss_cls,'smoothL1':smoothL1})
pretrained_model = tf.keras.applications.ResNet50(include_top=False)
frcnn_model = tf.keras.models.load_model('%s/faster_rcnn/frcnn/frcnn_model_weights.hdf5' % solution_dir)

ROI_POOL_SIZE = 3
PRE_NMS_PREDICT = 2048
POST_NMS_PREDICT = 128

def rpnWithPostProcess(filename):
    
    curr_img=tf.keras.utils.load_img(filename)
    curr_img = tf.keras.utils.img_to_array(curr_img)
    curr_img = np.expand_dims(curr_img, axis=0)

    curr_img = tf.keras.applications.resnet.preprocess_input(curr_img)

    im_width = curr_img.shape[2]
    im_height = curr_img.shape[1]

    feature_map = pretrained_model.predict(curr_img)

    feature_map=np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant')
    res =  rpn_model.predict(feature_map)
    objectness = np.swapaxes(res[0][0],0,1)
    boxregressions = np.swapaxes(res[1][0],0,1)
    stridex = im_width/np.shape(objectness)[0]
    stridey = im_height/np.shape(objectness)[1]
    h = np.shape(objectness)[0]
    w = np.shape(objectness)[1]

    all_anchors = getAllAnchors(im_width, im_height, stridex, stridey)

    objectness=objectness.reshape(-1,1)
    boxregressions=np.reshape(boxregressions,(-1,4))

    sort_indicies = objectness.ravel().argsort()

    proposals = bbox_transform_inv(all_anchors, boxregressions)[sort_indicies]
    proposals = proposals[proposals.shape[0]-PRE_NMS_PREDICT:proposals.shape[0]]

    threadsperblock = 6
    blockspergrid = ((proposals.shape[0]) + (threadsperblock - 1)) // threadsperblock
    nms_proposals = np.zeros((proposals.shape[0], 4), dtype=np.float32)
    non_max_suppression_gpu[blockspergrid, threadsperblock](proposals, nms_proposals)
    nms_inds = np.where(nms_proposals[:,2] > 0)
    nms_proposals = nms_proposals[nms_inds]

    if (len(nms_proposals) > POST_NMS_PREDICT):
        nms_proposals = nms_proposals[nms_proposals.shape[0]-POST_NMS_PREDICT:nms_proposals.shape[0]]

    for prop in nms_proposals:
        if(prop[0] - prop[2]/2 < 0):
            prop[2] = prop[0]*2
        if(prop[1] - prop[3]/2 < 0):
            prop[3] = prop[1]*2
        if(prop[0] + prop[2]/2 > im_width):
            prop[2] = (im_width-prop[0])*2-1
        if(prop[1] + prop[3]/2 > im_height):
            prop[3] = (im_height-prop[1])*2-1

    nms_proposals_orig_coords = np.zeros((nms_proposals.shape[0], nms_proposals.shape[1]), dtype=np.float32)
    for nm in range(nms_proposals.shape[0]):
        nms_proposals_orig_coords[nm, :] = nms_proposals[nm, :]

    x1 = nms_proposals[:, 0] - nms_proposals[:, 2]/2
    y1 = nms_proposals[:, 1] - nms_proposals[:, 3]/2
    x2 = nms_proposals[:, 0] + nms_proposals[:, 2]/2
    y2 = nms_proposals[:, 1] + nms_proposals[:, 3]/2
    nms_proposals[:, 0] = x1
    nms_proposals[:, 1] = y1
    nms_proposals[:, 2] = x2
    nms_proposals[:, 3] = y2

    grids = np.zeros((nms_proposals.shape[0],3,3,feature_map.shape[3]))
    roi_maps_pooled = RoI_Align(feature_map[0], nms_proposals, stridex, stridey, 3, grids)

    # nms_proposals[:, 0] = np.int32(nms_proposals[:, 0] / stridex)
    # nms_proposals[:, 1] = np.int32(nms_proposals[:, 1] / stridey)
    # nms_proposals[:, 2] = np.int32(nms_proposals[:, 2] / stridex)
    # nms_proposals[:, 3] = np.int32(nms_proposals[:, 3] / stridey)

    # roi_maps_pooled = []

    # for p in range(nms_proposals.shape[0]):

    #     ymin_roi = np.int32(nms_proposals[p, 1])
    #     ymax_roi = np.int32(nms_proposals[p, 3])+1
    #     xmin_roi = np.int32(nms_proposals[p, 0])
    #     xmax_roi = np.int32(nms_proposals[p, 2])+1

    #     roi_fmap = feature_map[0, ymin_roi:ymax_roi, xmin_roi:xmax_roi, :]

    #     roi_fmap_pooled = np.zeros((ROI_POOL_SIZE,ROI_POOL_SIZE,feature_map.shape[3]), dtype=np.float32)

    #     width_step = roi_fmap.shape[1]/ROI_POOL_SIZE
    #     height_step = roi_fmap.shape[0]/ROI_POOL_SIZE

    #     for i in range(ROI_POOL_SIZE):
    #         for j in range(ROI_POOL_SIZE):
    #                 xmin = np.int32(i*width_step)
    #                 ymin = np.int32(j*height_step)

    #                 xmax = np.int32((i+1)*width_step)
    #                 ymax = np.int32((j+1)*height_step)

    #                 if(xmin == xmax):
    #                     xmax +=1
    #                 if(ymin == ymax):
    #                     ymax +=1
    #                 roi_fmap_pooled[i, j, :]  = np.max(roi_fmap[ymin:ymax, xmin:xmax, :], axis=(0,1))
    #     roi_maps_pooled.append(roi_fmap_pooled)    
    return roi_maps_pooled, nms_proposals_orig_coords

def detectImage(filename):
    test_roi_fmaps, test_nms_props = rpnWithPostProcess(filename)

    res = frcnn_model.predict(test_roi_fmaps)

    predicted_classes = res[0]
    predicted_odd = res[1]
    predicted_deltas = res[2]

    predicted_boxes = bbox_transform_inv(test_nms_props, predicted_deltas)

    spirals = 0
    ellipticals = 0

    cutf = 0.9
    cutr = 0.5

    for cl in predicted_classes:
        if (cl[1] > cutf and cl[1] > cl[2]):
            spirals +=1
        if (cl[2] > cutf and cl[2] > cl[1]):
            ellipticals +=1

    showresults = Image.open(filename)
    drawresults = ImageDraw.Draw(showresults)

    boxes_to_draw = []
    classes_to_draw = []
    odd_to_draw = []

    for i in range(len(predicted_boxes)):
        if(predicted_classes[i, 1] > cutf or predicted_classes[i, 2] > cutf):
            a = predicted_boxes[i]
            boxes_to_draw.append(a)
            classes_to_draw.append(predicted_classes[i])
            odd_to_draw.append(predicted_odd[i])
            # if(predicted_odd[i, 0] > cutr):
            #     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='purple')
            #     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='purple')
            #     drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='purple')
            #     drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='purple')
            #     drawresults.text((a[0]-a[2]/2, a[1]+a[3]/2+20), str(predicted_odd[i]), fill="cyan")
            # else:
            #     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
            #     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
            #     drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
            #     drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
            # drawresults.text((a[0]-a[2]/2-100, a[1]-a[3]/2-20), str(predicted_classes[i]), fill="yellow")

    boxes_to_draw = np.asarray(boxes_to_draw)
    classes_to_draw = np.asarray(classes_to_draw)
    odd_to_draw = np.asarray(odd_to_draw)

    max_class = np.max(classes_to_draw, axis=1)
    max_classes = max_class.argsort()

    boxes_to_draw = boxes_to_draw[max_classes]
    classes_to_draw = classes_to_draw[max_classes]
    odd_to_draw = odd_to_draw[max_classes]

    threadsperblock = 6
    blockspergrid = ((boxes_to_draw.shape[0]) + (threadsperblock - 1)) // threadsperblock
    nms_boxes_to_draw = np.zeros((boxes_to_draw.shape[0], 4), dtype=np.float32)
    non_max_suppression_gpu2[blockspergrid, threadsperblock](boxes_to_draw, nms_boxes_to_draw)
    nms_inds = np.where(nms_boxes_to_draw[:,2] > 0)
    nms_boxes_to_draw = nms_boxes_to_draw[nms_inds]
    classes_to_draw = classes_to_draw[nms_inds]
    odd_to_draw = odd_to_draw[nms_inds]

    galaxyResults = []

    for i in range(len(nms_boxes_to_draw)):
            a = nms_boxes_to_draw[i]
            if(odd_to_draw[i, 0] > cutr and classes_to_draw[i, 1] > classes_to_draw[i, 2]):
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green', width=3)
                drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green', width=3)
                galaxyResults.append([a,0,1])
                # drawresults.text((a[0]-a[2]/2, a[1]+a[3]/2+20), str(odd_to_draw[i]), fill="cyan")

            if(odd_to_draw[i, 0] <= cutr and classes_to_draw[i, 1] > classes_to_draw[i, 2]):
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='blue', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='blue', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='blue', width=3)
                drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='blue', width=3)
                galaxyResults.append([a,1,0])

            if(odd_to_draw[i, 0] <= cutr and classes_to_draw[i, 1] <= classes_to_draw[i, 2]):
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red', width=3)
                drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red', width=3)
                drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red', width=3)
                galaxyResults.append([a,0,0])

            # drawresults.text((a[0]-a[2]/2-100, a[1]-a[3]/2-20), str(classes_to_draw[i]), fill="yellow")

    # showresults.show()
    showresults.save("%s/detector_gui/results/%s_FRCNN.jpg" % (solution_dir, filename.split("/")[-1].split(".")[0]))
    return showresults, galaxyResults

root = Tk()

img = ImageTk.PhotoImage(Image.open("%s/start_im.png" % project_dir))
panel = Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

global skyimagename
global galaxyResults

def saveGalaxies():
    global skyimagename
    for i in range(len(galaxyResults)):
        g = galaxyResults[i][0]
        loadedImage = Image.open(skyimagename)
        left = g[0]-g[2]/2
        top = g[1]-g[3]/2
        right = g[0]+g[2]/2
        bottom = g[1]+g[3]/2
        if ( left < 0):
            left = 0
        if ( right > loadedImage.width):
            right = loadedImage.width
        if ( top < 0):
            top = 0
        if ( bottom > loadedImage.height):
            bottom = loadedImage.height

        # print(left, top, right, bottom)

        imageNumber = skyimagename.split("/")[-1].split(".")[0]

        if ( left != right and top != bottom):
            crop = loadedImage.crop((left, top, right, bottom))
            # crop.save("%s/results/%s/%s.jpg" % (project_dir, imageNumber, i))
            # crop.save("%s/results/%s.jpg" % (project_dir, i))

def callback():
    global skyimagename
    skyimagename = fd.askopenfilename()
    loadedImage = Image.open(skyimagename)
    loadedImage = loadedImage.resize((1323,907))
    img2 = ImageTk.PhotoImage(loadedImage)
    panel.configure(image=img2)
    panel.image = img2

def detection():
    global galaxyResults
    results, galaxyResults = detectImage(skyimagename)
    results = results.resize((1323,907))
    img2 = ImageTk.PhotoImage(results)
    panel.configure(image=img2)
    panel.image = img2
    gc.collect()

button = Button(root,
                text="Open",
                command=lambda: callback())
button.pack(side=RIGHT)

detect = Button(root,
                text="Detect",
                command=lambda: detection())
detect.pack(side=RIGHT)

save = Button(root,
              text="Save",
              command=lambda: saveGalaxies())
save.pack(side=RIGHT)

root.mainloop()

