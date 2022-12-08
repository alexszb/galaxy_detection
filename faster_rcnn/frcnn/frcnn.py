from operator import truediv
from tracemalloc import get_traceback_limit
from xml.etree.ElementInclude import include
import numpy as np
import tensorflow as tf

from keras.models import Input, Model
from utils.util import loss_cls, smoothL1, getAllAnchors, bbox_transform_inv, bbox_transform, non_max_suppression_gpu, readBoxesFromFile, overlaps_gpu, RoI_Align

from PIL import Image, ImageDraw

import gc
import os
import pathlib

from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger
import time
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.main_class_loss = []
        self.odd_feature_loss = []
        self.main_acc = []
        self.odd_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.main_class_loss.append(logs.get('main_class_loss'))
        self.odd_feature_loss.append(logs.get('odd_feature_loss'))
        self.main_acc.append(logs.get("main_class_accuracy"))
        self.odd_acc.append(logs.get("odd_feature_accuracy"))

project_dir = os.path.dirname(__file__)
p = pathlib.Path(project_dir)
solution_dir = p.parent.parent

ROI_POOL_SIZE = 3
PRE_NMS_PREDICT = 2048
POST_NMS_PREDICT = 128

pretrained_model = tf.keras.applications.ResNet50(include_top=False)

base_img=tf.keras.utils.load_img("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, 50))
base_img = tf.keras.utils.img_to_array(base_img)
base_img = np.expand_dims(base_img, axis=0)

base_img = tf.keras.applications.resnet.preprocess_input(base_img)
base_feature_map=pretrained_model.predict(base_img)

stridex = base_img.shape[2]/np.shape(base_feature_map)[2]
stridey = base_img.shape[1]/np.shape(base_feature_map)[1]
all_anchors = getAllAnchors(base_img.shape[2], base_img.shape[1], stridex, stridey)

del base_img
del base_feature_map

rpn_model = tf.keras.models.load_model('%s/faster_rcnn/rpn/rpn_model.hdf5' % solution_dir, custom_objects={'loss_cls': loss_cls,'smoothL1':smoothL1})

# RCNN model

# roi_fmaps = Input(shape=(None,None,2048))

roi_f = Input(batch_shape=(None,ROI_POOL_SIZE,ROI_POOL_SIZE,2048))

flat = tf.keras.layers.Flatten()(roi_f)

fc= tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="fc"
    )(flat)

fc=tf.keras.layers.BatchNormalization()(fc)

dropoutlayer = tf.keras.layers.Dropout(
    0.25
)(fc)

output_deltas = tf.keras.layers.Dense(
        units=4,
        activation="linear",
        kernel_initializer="zeros",
        name="deltas"
    )(dropoutlayer)

output_main_class = tf.keras.layers.Dense(
        units=3,
        activation="softmax",
        kernel_initializer="zeros",
        name="main_class"
    )(dropoutlayer)

output_odd_feature = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_initializer="zeros",
        name="odd_feature"
    )(dropoutlayer)

model=Model(inputs=[roi_f],outputs=[output_main_class, output_odd_feature, output_deltas])
# model.summary()
model.compile(metrics='accuracy', optimizer='adam', loss={'deltas':'huber', 'main_class':'categorical_crossentropy', 'odd_feature': 'binary_crossentropy'})


def rpnWithPostProcess(filename):
    
    curr_img=tf.keras.utils.load_img("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, filename))
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

    # make boxes to be within image
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

    # region test nms proposals at prediction
    # showanchors = Image.open("%s/data/train/%s.jpg " % (solution_dir, filename))
    # draw = ImageDraw.Draw(showanchors)

    # for i in range(len(nms_proposals_orig_coords)):
    #     a = nms_proposals_orig_coords[i]
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    # showanchors.show()
    # endregion

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

def getBatch(filename, train_val_test):
    if ( train_val_test == 0):
        curr_img=tf.keras.utils.load_img("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, filename))
    if ( train_val_test == 1):
        curr_img=tf.keras.utils.load_img("%s/faster_rcnn/data/val/%s.jpg " % (solution_dir, filename))
    if ( train_val_test == 2):
        curr_img=tf.keras.utils.load_img("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, filename))
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

    objectness=objectness.reshape(-1,1)
    boxregressions=np.reshape(boxregressions,(-1,4))

    sort_indicies = objectness.ravel().argsort()
    # boxRegSorted = objectness[sort_indicies]

    proposals = bbox_transform_inv(all_anchors, boxregressions)[sort_indicies]
    proposals = proposals[proposals.shape[0]-2048:proposals.shape[0]]
    
    # nms_proposals = non_max_suppression(proposals)
    threadsperblock = 6
    blockspergrid = ((proposals.shape[0]) + (threadsperblock - 1)) // threadsperblock
    nms_proposals = np.zeros((proposals.shape[0], 4), dtype=np.float32)
    non_max_suppression_gpu[blockspergrid, threadsperblock](proposals, nms_proposals)
    nms_inds = np.where(nms_proposals[:,2] > 0)
    nms_proposals = nms_proposals[nms_inds]

    if (len(nms_proposals) > 512):
        nms_proposals = nms_proposals[nms_proposals.shape[0]-512:nms_proposals.shape[0]]

    # make boxes to be within image
    for prop in nms_proposals:
        if(prop[0] - prop[2]/2 < 0):
            prop[2] = prop[0]*2
        if(prop[1] - prop[3]/2 < 0):
            prop[3] = prop[1]*2
        if(prop[0] + prop[2]/2 > im_width):
            prop[2] = (im_width-prop[0])*2-1
        if(prop[1] + prop[3]/2 > im_height):
            prop[3] = (im_height-prop[1])*2-1

    # region test nms on image
    if(train_val_test == 0):
        showanchors = Image.open("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, filename))
    if(train_val_test == 1):
        showanchors = Image.open("%s/faster_rcnn/data/val/%s.jpg " % (solution_dir, filename))
    if(train_val_test == 2):
        showanchors = Image.open("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, filename))
    draw = ImageDraw.Draw(showanchors)

    # for i in range(len(nms_proposals)):
    #     a = nms_proposals[i]
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    # showanchors.show()
    # endregion

    if(train_val_test == 0):
        gt_boxes = readBoxesFromFile("%s/faster_rcnn/data/train/%s.txt" % (solution_dir, filename))
    if(train_val_test == 1):
        gt_boxes = readBoxesFromFile("%s/faster_rcnn/data/val/%s.txt" % (solution_dir, filename))
    if(train_val_test == 2):
        gt_boxes = readBoxesFromFile("%s/faster_rcnn/data/test/%s.txt" % (solution_dir, filename))    
    # region show gt boxes
    # for i in range(len(gt_boxes)):
    #     a = gt_boxes[i]
    #     if (a[4] == 1 and a[5] == 0):
    #         draw.line((a[0], a[1], a[0], a[3]),fill='blue', width=3)
    #         draw.line((a[2], a[1], a[2], a[3]),fill='blue', width=3)
    #         draw.line((a[0], a[1], a[2], a[1]),fill='blue', width=3)
    #         draw.line((a[0], a[3], a[2], a[3]),fill='blue', width=3)
    #     if (a[4] == 2 and a[5] == 0):
    #         draw.line((a[0], a[1], a[0], a[3]),fill='red', width=3)
    #         draw.line((a[2], a[1], a[2], a[3]),fill='red', width=3)
    #         draw.line((a[0], a[1], a[2], a[1]),fill='red', width=3)
    #         draw.line((a[0], a[3], a[2], a[3]),fill='red', width=3)
    #     if (a[4] == 1 and a[5] == 1):
    #         draw.line((a[0], a[1], a[0], a[3]),fill='green', width=3)
    #         draw.line((a[2], a[1], a[2], a[3]),fill='green', width=3)
    #         draw.line((a[0], a[1], a[2], a[1]),fill='green', width=3)
    #         draw.line((a[0], a[3], a[2], a[3]),fill='green', width=3)
    #     if (a[4] == 2 and a[5] == 1):
    #         draw.line((a[0], a[1], a[0], a[3]),fill='cyan', width=3)
    #         draw.line((a[2], a[1], a[2], a[3]),fill='cyan', width=3)
    #         draw.line((a[0], a[1], a[2], a[1]),fill='cyan', width=3)
    #         draw.line((a[0], a[3], a[2], a[3]),fill='cyan', width=3)
    # showanchors.show()
    # endregion

    threadsperblock = 2
    blockspergrid = (len(nms_proposals) + (threadsperblock - 1)) // threadsperblock
    overlaps = np.zeros(((len(nms_proposals)), gt_boxes.shape[0]), dtype=np.float32)
    overlaps_gpu[blockspergrid, threadsperblock](np.asarray(nms_proposals), gt_boxes, overlaps)

    #which gt_box closest to proposal
    argmax_overlaps = overlaps.argmax(axis=1)

    max_overlaps = overlaps[np.arange(len(nms_proposals)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    labels = np.empty((len(nms_proposals), ), dtype=np.float32)
    labels.fill(-1)
    # labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= .5] = 1
    labels[max_overlaps <= .05] = 0
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]

    coordinatesfg = np.asarray(nms_proposals)[fg_inds]
    if (len(fg_inds) > 32):
            disable_inds = np.random.choice(fg_inds, size=len(fg_inds)-32, replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels == 1)[0]

    if len(bg_inds) > len(fg_inds):
            disable_inds = np.random.choice(bg_inds, size=len(bg_inds)-(64-len(fg_inds)), replace=False)
            labels[disable_inds] = -1
            bg_inds = np.where(labels == 0)[0]

    batch_proposals=np.asarray(nms_proposals)[np.where(labels != -1)[0]]

    pos_boxes = nms_proposals[fg_inds]

    bb = bbox_transform(pos_boxes, gt_boxes[argmax_overlaps, :][fg_inds])

    # region test selected gt boxes for foreground proposals
    # showanchors = Image.open("%s/data/train/%s.jpg " % (solution_dir, filename))
    # draw = ImageDraw.Draw(showanchors)
    # for test_box in range(len(pos_boxes)):
        # a = gt_boxes[argmax_overlaps, :][fg_inds][test_box]
        # draw.line((a[0], a[1], a[0], a[3]),fill='white')
        # draw.line((a[2], a[1], a[2], a[3]),fill='white')
        # draw.line((a[0], a[1], a[2], a[1]),fill='white')
        # draw.line((a[0], a[3], a[2], a[3]),fill='white')

    #     a = nms_proposals[fg_inds][test_box]
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    #     draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')

    # for test_box in range(len(bg_inds)):
    #     a = nms_proposals[bg_inds][test_box]
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='yellow')
    #     draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='yellow')
    #     draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')
    #     draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')

    # showanchors.show()
    # endregion

    fg_which_box = argmax_overlaps[fg_inds]

    target_classes = np.zeros((nms_proposals.shape[0], 3), dtype=np.int32)
    target_odd_features = np.zeros((nms_proposals.shape[0], 1), dtype=np.int32)

    fg_classes = (gt_boxes[:, 4][fg_which_box])
    has_ring = gt_boxes[:, 5][fg_which_box]

    bbox_targets = np.zeros((nms_proposals.shape[0], 4), dtype=np.float32)
    for p in range(len(fg_inds)):
        if(fg_classes[p] == 1 or fg_classes[p] == 2):
            bbox_targets[fg_inds[p]] = bb[p]

    for bg in range(len(bg_inds)):
        target_classes[bg_inds[bg], 0] = 1
    
    bbox_targets = bbox_targets[np.where(labels != -1)[0]]

    for c in range(len(fg_classes)):
        if(fg_classes[c] == 1):
            target_classes[fg_inds[c],1] = 1
        if(fg_classes[c] == 2):
            target_classes[fg_inds[c],2] = 1

    for o in range(len(has_ring)):
        if(has_ring[o] == 1):
            target_odd_features[fg_inds[o],0] = 1

    target_classes = target_classes[np.where(labels != -1)[0]]
    target_odd_features = target_odd_features[np.where(labels != -1)[0]]

    # print(len(target_classes))
    # print(len(bbox_targets))

    # convert  proposal coordinates from xywh to x1y1x2y2
    x1 = batch_proposals[:, 0] - batch_proposals[:, 2]/2
    y1 = batch_proposals[:, 1] - batch_proposals[:, 3]/2
    x2 = batch_proposals[:, 0] + batch_proposals[:, 2]/2
    y2 = batch_proposals[:, 1] + batch_proposals[:, 3]/2
    batch_proposals[:, 0] = x1
    batch_proposals[:, 1] = y1
    batch_proposals[:, 2] = x2
    batch_proposals[:, 3] = y2

    # roi_maps_pooled = []

    # start = time.time()

    # for p in range(batch_proposals.shape[0]):
    #     regio = batch_proposals[p,:]
    #     regio[0] = regio[0]/stridex
    #     regio[1] = regio[1]/stridey
    #     regio[2] = regio[2]/stridex
    #     regio[3] = regio[3]/stridey
    #     grids = np.zeros((3,3,feature_map.shape[3]))
    #     roi_map = RoI_Align(feature_map[0], regio, 3, grids)
    #     roi_maps_pooled.append(roi_map)

    grids = np.zeros((batch_proposals.shape[0],3,3,feature_map.shape[3]))
    roi_maps_pooled = RoI_Align(feature_map[0], batch_proposals, stridex, stridey, 3, grids)

    # print(time.time()-start)
    # print(roi_maps_pooled[0])

    # roi_maps_pooled = []

    # batch_proposals[:, 0] = np.int32(batch_proposals[:, 0] / stridex)
    # batch_proposals[:, 1] = np.int32(batch_proposals[:, 1] / stridey)
    # batch_proposals[:, 2] = np.int32(batch_proposals[:, 2] / stridex)
    # batch_proposals[:, 3] = np.int32(batch_proposals[:, 3] / stridey)

    # for p in range(batch_proposals.shape[0]):

    #     ymin_roi = np.int32(batch_proposals[p, 1])
    #     ymax_roi = np.int32(batch_proposals[p, 3])+1
    #     xmin_roi = np.int32(batch_proposals[p, 0])
    #     xmax_roi = np.int32(batch_proposals[p, 2])+1

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

    return roi_maps_pooled, target_classes.tolist(), target_odd_features.tolist(), bbox_targets.tolist()


def val_input_generator():
    while (True):
        batch_rois = []
        batch_cls = []
        batch_bbx = []
        batch_odd = []
        for k in range(640,704):
            batchData= np.load("%s/rcnnTrainArrays/batchArray%s.npz" % (project_dir, str(int(k))))
            # roi_fmaps, cls_targets, box_targets = getBatch(k)
            roi_fmaps = batchData['arr1']
            cls_targets = batchData['arr2']
            odd_targets = batchData['arr3']
            box_targets = batchData['arr4']
            ringed = np.asarray(np.where(odd_targets == 1))[0][:32]
            non_ringed = np.asarray(np.where(odd_targets == 0))[0][:(64-ringed.shape[0])]
            balance = np.hstack((ringed, non_ringed))

            for z in balance:
                batch_rois.append(roi_fmaps[z])
                batch_cls.append(cls_targets[z])
                batch_bbx.append(box_targets[z])
                batch_odd.append(odd_targets[z])
                if(len(batch_rois)==64):
                    a=np.asarray(batch_rois)
                    b=np.asarray(batch_cls)
                    c=np.asarray(batch_bbx)
                    d=np.asarray(batch_odd)
                    yield a, [b, d, c]
                    batch_rois = []
                    batch_cls = []
                    batch_bbx = []
                    batch_odd = []
            gc.collect()

def input_generator():
    # save batches

    # for k in range(640):
    #     print(k)
    #     tot_rois = []
    #     tot_cls = []
    #     tot_bbx = []
    #     tot_odd = []
    #     for im in range(4):
    #         roi_fmaps, cls_targets, odd_targets, box_targets = getBatch(k*4+im, 0)
    #         for i in range(len(roi_fmaps)):
    #                 tot_rois.append(roi_fmaps[i])
    #                 tot_cls.append(cls_targets[i])
    #                 tot_bbx.append(box_targets[i])
    #                 tot_odd.append(odd_targets[i])
    #     np.savez_compressed("%s/rcnnTrainArrays/batchArray%s.npz" % (project_dir, str(int(k))), arr1 = np.asarray(tot_rois), arr2 = np.asarray(tot_cls), arr3 = np.asarray(tot_odd), arr4 = np.asarray(tot_bbx))
    #     gc.collect()

    # for k in range(64):
    #     print(640+k)
    #     tot_rois = []
    #     tot_cls = []
    #     tot_bbx = []
    #     tot_odd = []
    #     for im in range(4):
    #         roi_fmaps, cls_targets, odd_targets, box_targets = getBatch((640+k)*4+im, 1)
    #         for i in range(len(roi_fmaps)):
    #                 tot_rois.append(roi_fmaps[i])
    #                 tot_cls.append(cls_targets[i])
    #                 tot_bbx.append(box_targets[i])
    #                 tot_odd.append(odd_targets[i])
    #     np.savez_compressed("%s/rcnnTrainArrays/batchArray%s.npz" % (project_dir, str(int(640+k))), arr1 = np.asarray(tot_rois), arr2 = np.asarray(tot_cls), arr3 = np.asarray(tot_odd), arr4 = np.asarray(tot_bbx))
    #     gc.collect()

    while (True):
        batch_rois = []
        batch_cls = []
        batch_bbx = []
        batch_odd = []
        for k in range(0,640):
            batchData= np.load("%s/rcnnTrainArrays/batchArray%s.npz" % (project_dir, str(int(k))))
            # roi_fmaps, cls_targets, box_targets = getBatch(k)
            roi_fmaps = batchData['arr1']
            cls_targets = batchData['arr2']
            odd_targets = batchData['arr3']
            box_targets = batchData['arr4']
            
            indices = np.arange(roi_fmaps.shape[0])
            np.random.shuffle(indices)

            roi_fmaps = roi_fmaps[indices]
            cls_targets = cls_targets[indices]
            box_targets = box_targets[indices]
            odd_targets = odd_targets[indices]

            ringed = np.asarray(np.where(odd_targets == 1))[0][:32]
            non_ringed = np.asarray(np.where(odd_targets == 0))[0][:(64-ringed.shape[0])]
            balance = np.hstack((ringed, non_ringed))

            for z in balance:
                batch_rois.append(roi_fmaps[z])
                batch_cls.append(cls_targets[z])
                batch_bbx.append(box_targets[z])
                batch_odd.append(odd_targets[z])
                if(len(batch_odd)==64):
                    a=np.asarray(batch_rois)
                    b=np.asarray(batch_cls)
                    c=np.asarray(batch_bbx)
                    d=np.asarray(batch_odd)

                    # print(np.asarray(np.where(d == 1)).shape)
                    # print(np.asarray(np.where(d == 0)).shape)
                    # print(np.asarray(np.where(b[:,0] == 1)).shape)
                    # print(np.asarray(np.where(b[:,1] == 1)).shape)
                    # print(np.asarray(np.where(b[:,2] == 1)).shape)
                    yield a, [b, d, c]
                    batch_rois = []
                    batch_cls = []
                    batch_bbx = []
                    batch_odd = []
            gc.collect()

def lr_schedule(epoch):
    lrate = 0.0001*(1-0.9*epoch/10)
    return lrate


if not os.path.exists("%s/rcnnTrainArrays" % project_dir):
    os.makedirs("%s/rcnnTrainArrays" % project_dir)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./frcnn_test_chk_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
# reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=1, verbose=0)
# earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="auto", baseline=None,restore_best_weights=False)
# history = LossHistory()
# csv_logger = CSVLogger('log_epoch.csv', append=True, separator=';')


# model.fit(input_generator(), batch_size=64, steps_per_epoch=640, epochs=2, validation_data=val_input_generator(), validation_batch_size=64, validation_steps=64, callbacks=[checkpointer, tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

# with open('log.txt', 'w') as f:
#     for i in range(len(history.losses)):
#         f.write("%s,%s,%s,%s,%s\n" % (history.losses[i],history.main_class_loss[i], history.odd_feature_loss[i], history.main_acc[i],history.odd_acc[i]))

# model.save_weights('%s/frcnn_test_weights.hdf5' % project_dir) 

# # testing

model.load_weights("%s/frcnn_weights.hdf5" % project_dir)
# model.save("%s/frcnn_model_weights.hdf5" % project_dir)

# plot_model(model, to_file='%s/model_plot.png' % project_dir, show_shapes=True, show_layer_names=True, show_layer_activations=True)

test_img_number = 2816
test_roi_fmaps, test_nms_props = rpnWithPostProcess(test_img_number)

# region evaluate
def eval_gen():
    for z in range(2816,3072):
        a,b,c,d = getBatch(z, 2)
        yield a, [np.asarray(b), np.asarray(c), np.asarray(d)]

# model.evaluate(eval_gen(), batch_size=128, steps=256, verbose=1)
# endregion

res = model.predict(test_roi_fmaps)

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

print(spirals, ellipticals)

# region check delta prediction
# gt_boxes = readBoxesFromFile("%s/data/train/%s.txt" % (solution_dir, 20))

# for r in range (len(predicted_boxes)):
#     showresults = Image.open("%s/data/train/%s.jpg " % (solution_dir, 20))
#     drawresults = ImageDraw.Draw(showresults)

#     for i in range(len(gt_boxes)):
#             a = gt_boxes[i]
#             drawresults.line((a[0], a[1], a[0], a[3]),fill='white')
#             drawresults.line((a[2], a[1], a[2], a[3]),fill='white')
#             drawresults.line((a[0], a[1], a[2], a[1]),fill='white')
#             drawresults.line((a[0], a[3], a[2], a[3]),fill='white')
#     a = predicted_boxes[r]
#     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
#     drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
#     drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
#     drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
    
#     showresults.show()
# endregion

# gt_boxes = readBoxesFromFile("%s/data/train/%s.txt" % (solution_dir, test_img_number))
showresults = Image.open("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, test_img_number))
drawresults = ImageDraw.Draw(showresults)
# a = predicted_boxes[1]
# drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
# drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
# drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
# drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
# a = test_nms_props[1]
# drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='yellow')
# drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='yellow')
# drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')
# drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')

for i in range(len(predicted_boxes)):
    if(predicted_classes[i, 1] > cutf or predicted_classes[i, 2] > cutf):
        a = predicted_boxes[i]
        if(predicted_odd[i, 0] > cutr):
            drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='purple')
            drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='purple')
            drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='purple')
            drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='purple')
            drawresults.text((a[0]-a[2]/2, a[1]+a[3]/2+20), str(predicted_odd[i]), fill="cyan")
        else:
            drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
            drawresults.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
            drawresults.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
            drawresults.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
        drawresults.text((a[0]-a[2]/2-100, a[1]-a[3]/2-20), str(predicted_classes[i]), fill="yellow")

# for i in range(len(gt_boxes)):
#         a = gt_boxes[i]
#         drawresults.line((a[0], a[1], a[0], a[3]),fill='white')
#         drawresults.line((a[2], a[1], a[2], a[3]),fill='white')
#         drawresults.line((a[0], a[1], a[2], a[1]),fill='white')
#         drawresults.line((a[0], a[3], a[2], a[3]),fill='white')

# showresults.show()

# showresults.save("%s/result.jpg" % project_dir)

def ROC_curve():
    y_in = np.empty((0), dtype=int)
    y_out = np.empty((0), dtype=int)

    # odds = []
    for z in range(2816,3072):
        a,b,c,d = getBatch(z, 2)
        y_in = np.hstack((y_in, np.asarray(c).flatten()))
        # ring = np.where(np.asarray(c).flatten() == 1)[0]
        # non_ring = np.where(np.asarray(c).flatten() == 0)[0][:ring.shape[0]]
        # balance = np.hstack((ring, non_ring))
        # roi = []
        # odd = []
        # for s in balance:
        #     odd.append(np.asarray(c).flatten()[s])
        #     roi.append(np.asarray(a)[s])

        # if len(roi) > 0: 
        res = model.predict(np.asarray(a))[1]
        # res = np.where(res > 0.5, 1, 0)
        y_out = np.hstack((y_out, res.flatten()))
            # odds.extend(odd)

    # RocCurveDisplay.from_predictions(np.asarray(odd), y_out)
    PrecisionRecallDisplay.from_predictions(y_in, y_out)
    # print(precision_score(y_in, y_out))
    # print(recall_score(y_in, y_out))
    # print(f1_score(y_in, y_out))
    plt.xlabel("felidézés")
    plt.ylabel("precizitás")
    plt.show()
    
ROC_curve()