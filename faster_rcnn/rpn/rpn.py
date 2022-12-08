import numba.cuda as cuda
import numpy as np
import cupy as cupy

import random
import math
import os
import pathlib
import time

from PIL import Image
from PIL import ImageDraw

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.layers import Conv2D
from keras.models import Input, Model

import keras.backend as K
from tensorflow.python.ops.gen_array_ops import fill
#from utils import bbx
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.scores_loss = []
        self.delta_loss = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.scores_loss.append(logs.get('scores_loss'))
        self.delta_loss.append(logs.get('deltas_loss'))

epoch_number = 10

project_dir = os.path.dirname(__file__)
p = pathlib.Path(project_dir)
solution_dir = p.parent.parent

showanchors = Image.open("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, 2816))

draw = ImageDraw.Draw(showanchors)
#pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(1361,1984,3))

pretrained_model = tf.keras.applications.ResNet50(include_top=False)

### RPN model
def loss_cls(y_true, y_pred):
    
    condition = K.not_equal(y_true, -1)
    indices = K.tf.where(condition)
    target = K.tf.gather_nd(y_true, indices)
    output = K.tf.gather_nd(y_pred, indices)
    loss = K.binary_crossentropy(target, output)
    return loss

def smoothL1(y_true, y_pred):
    nd=K.tf.where(K.tf.not_equal(y_true,0))
    target=K.tf.gather_nd(y_true,nd)
    output=K.tf.gather_nd(y_pred,nd)
    x = tf.compat.v1.losses.huber_loss(target,output, delta = 1.0, )
    return K.mean(x)

feature_map_tile = Input(shape=(None,None,2048))
convolution_3x3 = Conv2D(
    filters=512,
    kernel_size=(3, 3),
    activation='relu',
    name="3x3"
)(feature_map_tile)

output_deltas = Conv2D(
    filters= 4 * 9,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="zeros",
    name="deltas"
)(convolution_3x3)

output_scores = Conv2D(
    filters=9,
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="scores"
)(convolution_3x3)

model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
model.compile(optimizer='adam', loss={'scores':loss_cls, 'deltas':smoothL1})

def getAnchorBoxes(anchor, scales):
    anchorBoxes = np.vstack([
            [0,0,anchor[2]*scales[0],anchor[3]*scales[0]],
            [0,0,anchor[2]*scales[1],anchor[3]*scales[1]],
            [0,0,anchor[2]*scales[2],anchor[3]*scales[2]],
            [0,0,anchor[2]*scales[0]*2,anchor[3]*scales[0]*0.5],
            [0,0,anchor[2]*scales[1]*2,anchor[3]*scales[1]*0.5],
            [0,0,anchor[2]*scales[2]*2,anchor[3]*scales[2]*0.5],
            [0,0,anchor[2]*scales[0]*0.5,anchor[3]*scales[0]*2],
            [0,0,anchor[2]*scales[1]*0.5,anchor[3]*scales[1]*2],
            [0,0,anchor[2]*scales[2]*0.5,anchor[3]*scales[2]*2]
        ])
    return anchorBoxes

def getAllAnchors(width, height, stridex, stridey):
    scales = [3,4,5]
    anchors = getAnchorBoxes((0, 0, stridex, stridex), scales)
    x_centers = np.arange(0+stridex/2,(width-stridex/2)+1,stridex)
    y_centers = np.arange(0+stridey/2,(height-stridey/2)+1,stridey)

    center_list = np.array(np.meshgrid(x_centers, y_centers, sparse=False, indexing='xy',)).T.reshape(-1,2)
    anchorsSizes = anchors[:,np.arange(2,4)]
    anchorsCenters = anchors[:,np.arange(0,2)]

    all_anchorsstack = np.empty((0,4), int)
    for i in center_list:
        newAnchor = np.hstack([i+anchorsCenters, anchorsSizes])
        all_anchorsstack = np.vstack((all_anchorsstack, newAnchor))

    return all_anchorsstack

base_img=tf.keras.utils.load_img("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, 0))
base_img = tf.keras.utils.img_to_array(base_img)
base_img = np.expand_dims(base_img, axis=0)

base_img = tf.keras.applications.resnet.preprocess_input(base_img)
feature_map=pretrained_model.predict(base_img)

stridex = base_img.shape[2]/np.shape(feature_map)[2]
stridey = base_img.shape[1]/np.shape(feature_map)[1]
all_anchors = getAllAnchors(base_img.shape[2], base_img.shape[1], stridex, stridey)

def readBoxesFromFile(filename):
    gt_boxes = np.empty((0,4), int)
    with open(filename) as openfileobject:
        for line in openfileobject:
            data = line.split(',')
            gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4])]
            gt_boxes = np.vstack((gt_boxes,gt_box))
    return gt_boxes

def bbox_overlaps(anchors, gt_boxes):
    overlaps = np.zeros((len(anchors),len(gt_boxes)), dtype=float)
    for a in range(len(anchors)):
        anchor = anchors[a]
        anchor_xmin = anchor[0]-anchor[2]/2
        anchor_ymin = anchor[1]-anchor[3]/2
        anchor_xmax = anchor[0]+anchor[2]/2
        anchor_ymax = anchor[1]+anchor[3]/2

        for g in range(len(gt_boxes)):
            gt_box = gt_boxes[g]
            dx = 0
            if (gt_box[0] >= anchor_xmin and gt_box[2] <= anchor_xmax):
                dx = gt_box[2] - gt_box[0]
            if (gt_box[0] >= anchor_xmin and gt_box[0] <= anchor_xmax and gt_box[2] >= anchor_xmax):
                dx = anchor_xmax - gt_box[0]
            if (gt_box[0] <= anchor_xmin and gt_box[2] <= anchor_xmax and gt_box[2] >= anchor_xmin):
                dx = gt_box[2] - anchor_xmin
            if (gt_box[0] <= anchor_xmin and gt_box[2] >= anchor_xmax):
                dx = anchor_xmax  - anchor_xmin
            dy = 0
            if (gt_box[1] >= anchor_ymin and gt_box[3] <= anchor_ymax):
                dy = gt_box[3] - gt_box[1]
            if (gt_box[1] >= anchor_ymin and gt_box[1] <= anchor_ymax and gt_box[3] >= anchor_ymax):
                dy = anchor_ymax - gt_box[1]
            if (gt_box[1] <= anchor_ymin and gt_box[3] <= anchor_ymax and gt_box[3] >= anchor_ymin):
                dy = gt_box[3] - anchor_ymin
            if (gt_box[1] <= anchor_ymin and gt_box[3] >= anchor_ymax):
                dy = anchor_ymax  - anchor_ymin

            dx = dx + 1
            dy = dy + 1
            overlap = dx*dy
            area_of_anchor = ((anchor_xmax-anchor_xmin)+1)*((anchor_ymax-anchor_ymin)+1)
            area_of_gt = ((gt_box[2]-gt_box[0])+1) * ((gt_box[3] - gt_box[1])+1)
            area_of_union = area_of_anchor+area_of_gt-overlap

            iou = overlap/area_of_union
            overlaps[a, g] = iou
    return overlaps

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2]
    ex_heights = ex_rois[:, 3]
    ex_ctr_x = ex_rois[:, 0]
    ex_ctr_y = ex_rois[:, 1]

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = np.transpose(targets)

    return targets

### test faster method
def faster_bbox_overlaps(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float16)
    for k in range(K):
        for n in range(N):
            iw = cupy.minimum((boxes[n,0]+(boxes[n,2]/2)), query_boxes[k,2]) - cupy.maximum((boxes[n,0]-(boxes[n,2]/2)), query_boxes[k,0]+1)
            if iw > 0:
                ih = cupy.minimum((boxes[n,1]+(boxes[n,3]/2)), query_boxes[k,3]) - cupy.maximum((boxes[n,1]-(boxes[n,3]/2)), query_boxes[k,1]+1)
                if ih > 0:
                    ua = ((((boxes[n,0]+(boxes[n,2]/2)) - (boxes[n,0]-(boxes[n,2]/2))) + 1) * (((boxes[n,1]+(boxes[n,3]/2)) - (boxes[n,1]-(boxes[n,3]/2))) + 1) + (((query_boxes[k,2] - query_boxes[k,0] + 1) * (query_boxes[k,3] - query_boxes[k,1] + 1)) - iw * ih))
                    overlaps[n, k] = iw*ih / ua
    return overlaps

@cuda.jit
def overlaps_gpu(anchor_boxes, gt_boxes, overlaps):
    for k in range(gt_boxes.shape[0]):
        box_area = ((gt_boxes[k, 2] - gt_boxes[k, 0] + 1) * (gt_boxes[k, 3] - gt_boxes[k, 1] + 1))
        iw = (min(anchor_boxes[cuda.grid(1), 0]+(anchor_boxes[cuda.grid(1), 2]/2), gt_boxes[k, 2]) - max(anchor_boxes[cuda.grid(1), 0]-(anchor_boxes[cuda.grid(1), 2]/2), gt_boxes[k, 0]))+1
        if iw > 0:
            ih = (min(anchor_boxes[cuda.grid(1), 1]+(anchor_boxes[cuda.grid(1), 3]/2), gt_boxes[k, 3]) - max(anchor_boxes[cuda.grid(1), 1]-(anchor_boxes[cuda.grid(1), 3]/2), gt_boxes[k, 1]))+1
            if ih > 0:
                ua = ((((anchor_boxes[cuda.grid(1), 0]+anchor_boxes[cuda.grid(1), 2]/2)) - (anchor_boxes[cuda.grid(1), 0]-(anchor_boxes[cuda.grid(1), 2]/2))) + 1) * (((anchor_boxes[cuda.grid(1), 1]+(anchor_boxes[cuda.grid(1), 3]/2)) - (anchor_boxes[cuda.grid(1), 1]-(anchor_boxes[cuda.grid(1),3]/2))) + 1) + (box_area - iw * ih)
                overlaps[cuda.grid(1), k] = (iw * ih) / ua

def getBatch(filename, all_anchors):

    img=tf.keras.utils.load_img("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, filename))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.resnet.preprocess_input(img)
    feature_map=pretrained_model.predict(img)

    gt_boxes = readBoxesFromFile("%s/faster_rcnn/data/train/%s.txt" % (solution_dir, filename))
    # all_anchors = getAllAnchors(img.shape[2], img.shape[1], stridex, stridey)

    #region showanchors
    # showanchors = Image.open("%s/faster_rcnn/data/train/%s.jpg " % (solution_dir, filename))
    # draw = ImageDraw.Draw(showanchors)
    # for a in gt_boxes:  
    #             draw.line((a[0], a[1], a[0], a[3]),fill='white')
    #             draw.line((a[2], a[1], a[2], a[3]),fill='white')
    #             draw.line((a[0], a[1], a[2], a[1]),fill='white')
    #             draw.line((a[0], a[3], a[2], a[3]),fill='white')
    # showanchors.show()

    # img = Image.new(mode="RGB", size=(1984, 1361), color="black")
    # drawa = ImageDraw.Draw(img)
    # for a in all_anchors:
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
    #             draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
    #             draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
    #             draw.point((a[0], a[1]), fill="pink")
    # img.show()
    #endregion
    #print(all_anchors.shape)

    # overlaps = faster_bbox_overlaps(all_anchors, gt_boxes)
    # overlaps = bbx.bbox_overlaps(np.ascontiguousarray(all_anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))

    threadsperblock = 256
    blockspergrid = (all_anchors.shape[0] + (threadsperblock - 1)) // threadsperblock
    overlaps = np.zeros((all_anchors.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    overlaps_gpu[blockspergrid, threadsperblock](all_anchors, gt_boxes, overlaps)

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(all_anchors)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                    np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    labels = np.empty((len(all_anchors), ), dtype=np.float32)
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= .7] = 1
    labels[max_overlaps <= .3] = 0
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    # print(fg_inds)
    # print(len(fg_inds))

    # equal fg and bg with batch size 32, so 16-16
    # if len(fg_inds) > 16:
    #     disable_inds = np.random.choice(fg_inds, size=len(fg_inds)-16, replace=False)
    #     labels[disable_inds] = -1

    # if len(bg_inds) > len(fg_inds):
    #     disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - 16), replace=False)
    #     labels[disable_inds] = -1

    #dynamic size, equal fg and bg
    # if len(bg_inds) > len(fg_inds):
    #     disable_inds = np.random.choice(bg_inds, size=(len(bg_inds)- len(fg_inds)), replace=False)
    #     labels[disable_inds] = -1

    # padded with negatives to number
    if len(bg_inds) > len(fg_inds):
        disable_inds = np.random.choice(bg_inds, size=len(bg_inds)-(64-len(fg_inds)), replace=False)
        labels[disable_inds] = -1

    batch_inds=np.where(labels != -1)[0]
    

    pos_inds = np.where(labels == 1)[0]

    #region showcoordinates
    # coordinates = all_anchors[batch_inds]
    # for a in coordinates:  
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='yellow')
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='yellow')
    #             draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')
    #             draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='yellow')

    coordinatesfg = all_anchors[fg_inds]
    # for a in coordinatesfg:  
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
    #             draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
    #             draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    #             draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
    # showanchors.show()
    #endregion
    padded_fcmap=np.pad(feature_map,((0,0),(1,2),(1,2),(0,0)),mode='constant')
    padded_fcmap=np.squeeze(padded_fcmap)
    batch_inds = (batch_inds / 9).astype(np.int)
    batch_label_targets=labels.reshape(-1,1,1,1*9)[batch_inds]

    #region showfeaturemap
    # plt.figure(figsize=(20,20))
    # plt.imshow(feature_map[0,:,:,0])
    # plt.show()
    #endregion

    pos_anchors = all_anchors[pos_inds]
    bbox_targets = np.zeros((len(all_anchors), 4), dtype=np.float32)
    box_targets = gt_boxes[argmax_overlaps][labels == 1]
    bb = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][pos_inds])
    for p in range(len(pos_inds)):
        bbox_targets[pos_inds[p]] = bb[p]
    batch_bbox_targets = bbox_targets.reshape(-1,1,1,4*9)[batch_inds]

    batch_tiles = []

    for ind in batch_inds:
            y = ind % feature_map.shape[1]
            x = int(ind/(feature_map.shape[1]))
            fc_snip=padded_fcmap[y:y+3,x:x+3,:]
            batch_tiles.append(fc_snip)

    return batch_tiles, batch_label_targets.tolist(), batch_bbox_targets.tolist()

def val_input_generator():
    while True:
        batch_tls = []
        batch_lbs = []
        batch_bbx = []
        for im in range(80, 88):      
            # tls = np.load("trainArrays/tlsArray%s.npy" % str(int(im)))
            # lbs = np.load("trainArrays/lbsArray%s.npy" % str(int(im)))
            # bbx = np.load("trainArrays/bbxArray%s.npy" % str(int(im)))

            batchData= np.load("%s/trainArrays/batchArray%s.npz" % (project_dir, str(int(im))))
            tls = batchData['arr1']
            lbs = batchData['arr2']
            bbx = batchData['arr3']

            # batch_tls = []
            # batch_lbs = []
            # batch_bbx = []
            for z in range(2048):
                batch_tls.append(tls[z])
                batch_lbs.append(lbs[z])
                batch_bbx.append(bbx[z])
                if(len(batch_tls)==256):
                    a=np.asarray(batch_tls)
                    b=np.asarray(batch_lbs)
                    c=np.asarray(batch_bbx)
                            #print(a.shape)
                            #print(b.shape)
                    yield a, [b, c]
                    batch_tls=[]
                    batch_lbs=[]
                    batch_bbx=[]

def input_generator():
    tls=[]
    lbs=[]
    bbx=[]
    for im in range(80):
        print(im)
        for k in range(32):
            tiles, labels, bboxes = getBatch(im*32+k, all_anchors)
            for i in range(len(tiles)):
                tls.append(tiles[i])
                lbs.append(labels[i])
                bbx.append(bboxes[i])
        
        # np.save("%s/trainArrays/tlsArray%s.npy" % (project_dir, im), tls)
        # np.save("%s/trainArrays/lbsArray%s.npy" % (project_dir, im), lbs)
        # np.save("%s/trainArrays/bbxArray%s.npy" % (project_dir, im), bbx)

        np.savez_compressed("%s/trainArrays/batchArray%s.npz" % (project_dir, str(int(im))), arr1 = tls, arr2 = lbs, arr3 = bbx)
        tls=[]
        lbs=[]
        bbx=[]

    for im in range(8):
        for k in range(32):
            tiles, labels, bboxes = getBatch(im*32+k, all_anchors)
            for i in range(len(tiles)):
                    tls.append(tiles[i])
                    lbs.append(labels[i])
                    bbx.append(bboxes[i])
                    
        # np.save("%s/trainArrays/tlsArray%s.npy" % (project_dir, im), tls)
        # np.save("%s/trainArrays/lbsArray%s.npy" % (project_dir, im), lbs)
        # np.save("%s/trainArrays/bbxArray%s.npy" % (project_dir, im), bbx)

        np.savez_compressed("%s/trainArrays/batchArray%s.npz" % (project_dir, str(int(80+im))), arr1 = tls, arr2 = lbs, arr3 = bbx)
        tls=[]
        lbs=[]
        bbx=[]

    while True:
        batch_tls = []
        batch_lbs = []
        batch_bbx = []
        for im in range(80):      
            # tls = np.load("trainArrays/tlsArray%s.npy" % str(int(im)))
            # lbs = np.load("trainArrays/lbsArray%s.npy" % str(int(im)))
            # bbx = np.load("trainArrays/bbxArray%s.npy" % str(int(im)))

            batchData= np.load("%s/trainArrays/batchArray%s.npz" % (project_dir, str(int(im))))
            tls = batchData['arr1']
            lbs = batchData['arr2']
            bbx = batchData['arr3']

            indices = np.arange(tls.shape[0])
            np.random.shuffle(indices)

            tls = tls[indices]
            lbs = lbs[indices]
            bbx = bbx[indices]

            # batch_tls = []
            # batch_lbs = []
            # batch_bbx = []
            for z in range(2048):
                batch_tls.append(tls[z])
                batch_lbs.append(lbs[z])
                batch_bbx.append(bbx[z])
                if(len(batch_tls)==256):
                    a=np.asarray(batch_tls)
                    b=np.asarray(batch_lbs)
                    c=np.asarray(batch_bbx)
                            #print(a.shape)
                            #print(b.shape)
                    yield a, [b, c]
                    batch_tls=[]
                    batch_lbs=[]
                    batch_bbx=[]

def lr_schedule(epoch):
    lrate = 0.0001*(1-0.9*epoch/epoch_number)
    return lrate

if not os.path.exists("%s/trainArrays" % project_dir):
    os.makedirs("%s/trainArrays" % project_dir)

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='%s/chk_weights.hdf5' % project_dir, monitor='loss', verbose=1, save_best_only=True)
model.summary()
history = LossHistory()
csv_logger = CSVLogger('log_epoch.csv', append=True, separator=';')


model.fit(input_generator(), batch_size=256, steps_per_epoch = 640, validation_data=val_input_generator(), validation_batch_size=256, validation_steps=64, epochs=epoch_number, callbacks=[history, csv_logger, LearningRateScheduler(lr_schedule)])
model.save_weights('%s/weights.hdf5' % project_dir) 
# model.load_weights("%s/weights.hdf5" % project_dir)
model.save("%s/rpn_model.hdf5" % project_dir)

with open('log.txt', 'w') as f:
    for i in range(len(history.losses)):
        f.write("%s,%s,%s\n" % (history.losses[i], history.scores_loss[i], history.delta_loss[i]))

# plot_model(model, to_file='%s/model_plot.png' % project_dir, show_shapes=True, show_layer_names=True, show_layer_activations=True)

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2]
    heights = boxes[:, 3]
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)

    pred_boxes[:, 0::4] = pred_ctr_x
    pred_boxes[:, 1::4] = pred_ctr_y
    pred_boxes[:, 2::4] = pred_w
    pred_boxes[:, 3::4] = pred_h

    return pred_boxes

def bbox_transform_inv_single(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    widths = boxes[2]
    heights = boxes[3]
    ctr_x = boxes[0]
    ctr_y = boxes[1]

    dx = deltas[0]
    dy = deltas[1]
    dw = deltas[2]
    dh = deltas[3]

    pred_ctr_x = dx * widths+ ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(4, dtype=deltas.dtype)

    pred_boxes[0] = pred_ctr_x
    pred_boxes[1] = pred_ctr_y
    pred_boxes[2] = pred_w
    pred_boxes[3] = pred_h
    return pred_boxes

tst=tf.keras.utils.load_img("%s/faster_rcnn/data/test/%s.jpg " % (solution_dir, 2816))
tst = tf.keras.utils.img_to_array(tst)
tst = np.expand_dims(tst, axis=0)
test = tf.keras.applications.resnet.preprocess_input(tst)
feature_map=pretrained_model.predict(tst)
feature_map=np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant')
res = model.predict(feature_map)
x = res[0][0]
y = res[1][0]
stridex = 1984/np.shape(x)[1]
stridey = 1361/np.shape(x)[0]
h = np.shape(x)[0]
w = np.shape(x)[1]
count = 0
#print(x[0][0].shape)
np.set_printoptions(precision=1, suppress=True)

#results = Image.open("2.png")
#draw = ImageDraw.Draw(results)

# all_anchors = getAllAnchors(1984, 1361, stridex, stridey)
deltas=np.reshape(y,(-1,4))
# for a in all_anchors:
#                 draw.point((a[0], a[1]), fill="pink")
cutf = 0.995


for i in range(h):
    for j in range(w):
        numbers = x[i,j,:]
        haspos = False
        # print(numbers)
        #for n in numbers:
        if (numbers[0] > cutf or numbers[1] > cutf or numbers[2] > cutf or numbers[3] > cutf or numbers[4] > cutf or numbers[5] > cutf or numbers[6] > cutf):
                xk = j*stridex+16
                yk = i*stridey+16
                draw.point((xk, yk), fill='green')
                draw.point((xk+1, yk), fill='green')
                draw.point((xk-1, yk), fill='green')
                draw.point((xk, yk+1), fill='green')
                draw.point((xk, yk-1), fill='green')
                idx = (j*43+i)*9
                if(numbers[0] > cutf):
                    a = bbox_transform_inv_single(all_anchors[idx], y[i,j,0:4])
                    draw.point((a[0],a[1]), fill='pink')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
                    a = all_anchors[idx]
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
                if(numbers[1] > cutf):
                    a = bbox_transform_inv_single(all_anchors[idx+1], y[i,j,4:8])
                    draw.point((a[0],a[1]), fill='pink')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')    
                    a = all_anchors[idx+1]
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')      
                if(numbers[2] > cutf):
                    a = bbox_transform_inv_single(all_anchors[idx+2], y[i,j,8:12])
                    draw.point((a[0],a[1]), fill='pink')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
                    draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')  
                    a = all_anchors[idx+2]
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
                    # draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')       
               
                #print(j,i)
                haspos = True
        if haspos == True:
            count += 1
print(count)

# idx = (7*43+17)*9
# a = bbox_transform_inv_single(all_anchors[idx+2], y[17,7,8:12])
# draw.point((a[0],a[1]), fill='pink')
# draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
# draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
# draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
# draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')    

# a = all_anchors[idx+2]
# draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='green')
# draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='green')
# draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')
# draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='green')   

showanchors.show()

#results.show()

# scores = x
# scores=scores.reshape(-1,1)
# deltas=np.reshape(y,(-1,4))
# all_anchors = getAllAnchors(1984, 1361, stridex, stridey)
# print(len(all_anchors))
# proposals = bbox_transform_inv(all_anchors, deltas)

# for i in range(len(scores)):
#     if(scores[i] > 0.95):
#                 a = all_anchors[i]
#                 draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]-a[3]/2),fill='red')
#                 draw.line((a[0]-a[2]/2, a[1]-a[3]/2, a[0]-a[2]/2, a[1]+a[3]/2),fill='red')
#                 draw.line((a[0]-a[2]/2, a[1]+a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
#                 draw.line((a[0]+a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2),fill='red')
#showanchors.show()