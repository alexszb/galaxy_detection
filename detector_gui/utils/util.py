from operator import truediv
import numpy as np
import tensorflow as tf
import keras.backend as K
import numba.cuda as cuda
import math

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
    scales = [1,1.5,3]
    anchors = getAnchorBoxes((0, 0, stridex, stridex), scales)
    x_centers = np.arange(0+stridex/2,(width-stridex/2)+1,stridex)
    y_centers = np.arange(0+stridey/2,(height-stridey/2)+1,stridey)

    center_list =  np.array(np.meshgrid(x_centers, y_centers, sparse=False, indexing='xy',)).T.reshape(-1,2)
    anchorsSizes = anchors[:,np.arange(2,4)]
    anchorsCenters = anchors[:,np.arange(0,2)]

    all_anchorsstack = np.empty((0,4), int)
    for i in center_list:
        newAnchor = np.hstack([i+anchorsCenters, anchorsSizes])
        all_anchorsstack = np.vstack((all_anchorsstack, newAnchor))

    return all_anchorsstack

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
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
    # x
    pred_boxes[:, 0::4] = pred_ctr_x
    # y
    pred_boxes[:, 1::4] = pred_ctr_y
    # w
    pred_boxes[:, 2::4]= pred_w
    # h
    pred_boxes[:, 3::4] = pred_h
    return pred_boxes

@cuda.jit
def non_max_suppression_gpu(proposals, nms_proposals):
    discard = False
    for j in range(proposals.shape[0]):
        box_area = (((proposals[j, 0] + proposals[j, 2]/2) - (proposals[j, 0] - proposals[j, 2]/2)) + 1)* (((proposals[j, 1] + proposals[j, 3]/2) - (proposals[j, 1]-proposals[j, 3]/2)) + 1)
        iw = min(proposals[cuda.grid(1), 0] + proposals[cuda.grid(1), 2]/2, proposals[j, 0] + proposals[j, 2]/2) - max(proposals[cuda.grid(1), 0] -  proposals[cuda.grid(1), 2]/2, proposals[j, 0] - proposals[j, 2]/2) + 1
        if iw > 0:
            ih = min(proposals[cuda.grid(1), 1] + proposals[cuda.grid(1), 3]/2, proposals[j, 1] + proposals[j, 3]/2) - max(proposals[cuda.grid(1), 1] -  proposals[cuda.grid(1), 3]/2, proposals[j, 1] - proposals[j, 3]/2) + 1
            if ih > 0:
                ua = (((proposals[cuda.grid(1), 0] + proposals[cuda.grid(1), 2]/2) - (proposals[cuda.grid(1), 0] - proposals[cuda.grid(1), 2]/2))+1)*(((proposals[cuda.grid(1), 1]+proposals[cuda.grid(1),3]/2)-(proposals[cuda.grid(1),1]-proposals[cuda.grid(1),3]/2))+1) + (box_area - iw*ih)
                overlaps = (iw*ih) / ua
                if (overlaps > 0.7 and j > cuda.grid(1)):
                    discard = True
    if (not discard):
        nms_proposals[cuda.grid(1),0] = (proposals[cuda.grid(1),0])
        nms_proposals[cuda.grid(1),1] = (proposals[cuda.grid(1),1])
        nms_proposals[cuda.grid(1),2] = (proposals[cuda.grid(1),2])
        nms_proposals[cuda.grid(1),3] = (proposals[cuda.grid(1),3])

@cuda.jit
def non_max_suppression_gpu2(proposals, nms_proposals):
    discard = False
    for j in range(proposals.shape[0]):
        box_area = (((proposals[j, 0] + proposals[j, 2]/2) - (proposals[j, 0] - proposals[j, 2]/2)) + 1)* (((proposals[j, 1] + proposals[j, 3]/2) - (proposals[j, 1]-proposals[j, 3]/2)) + 1)
        iw = min(proposals[cuda.grid(1), 0] + proposals[cuda.grid(1), 2]/2, proposals[j, 0] + proposals[j, 2]/2) - max(proposals[cuda.grid(1), 0] -  proposals[cuda.grid(1), 2]/2, proposals[j, 0] - proposals[j, 2]/2) + 1
        if iw > 0:
            ih = min(proposals[cuda.grid(1), 1] + proposals[cuda.grid(1), 3]/2, proposals[j, 1] + proposals[j, 3]/2) - max(proposals[cuda.grid(1), 1] -  proposals[cuda.grid(1), 3]/2, proposals[j, 1] - proposals[j, 3]/2) + 1
            if ih > 0:
                ua = (((proposals[cuda.grid(1), 0] + proposals[cuda.grid(1), 2]/2) - (proposals[cuda.grid(1), 0] - proposals[cuda.grid(1), 2]/2))+1)*(((proposals[cuda.grid(1), 1]+proposals[cuda.grid(1),3]/2)-(proposals[cuda.grid(1),1]-proposals[cuda.grid(1),3]/2))+1) + (box_area - iw*ih)
                overlaps = (iw*ih) / ua
                if (overlaps > 0.3 and j > cuda.grid(1)):
                    discard = True
    if (not discard):
        nms_proposals[cuda.grid(1),0] = (proposals[cuda.grid(1),0])
        nms_proposals[cuda.grid(1),1] = (proposals[cuda.grid(1),1])
        nms_proposals[cuda.grid(1),2] = (proposals[cuda.grid(1),2])
        nms_proposals[cuda.grid(1),3] = (proposals[cuda.grid(1),3])


def non_max_suppression(proposals):
    nms_proposals = []
    for i in range(proposals.shape[0]):
        discard = False
        for j in range(proposals.shape[0]):
            box_area = (((proposals[j, 0] + proposals[j, 2]/2) - (proposals[j, 0] - proposals[j, 2]/2)) + 1)* (((proposals[j, 1] + proposals[j, 3]/2) - (proposals[j, 1]-proposals[j, 3]/2)) + 1)
            iw = min(proposals[i, 0] + proposals[i, 2]/2, proposals[j, 0] + proposals[j, 2]/2) - max(proposals[i, 0] -  proposals[i, 2]/2, proposals[j, 0] - proposals[j, 2]/2) + 1
            if iw > 0:
                ih = min(proposals[i, 1] + proposals[i, 3]/2, proposals[j, 1] + proposals[j, 3]/2) - max(proposals[i, 1] -  proposals[i, 3]/2, proposals[j, 1] - proposals[j, 3]/2) + 1
                if ih > 0:
                    ua = (((proposals[i, 0] + proposals[i, 2]/2) - (proposals[i, 0] - proposals[i, 2]/2))+1)*(((proposals[i, 1]+proposals[i,3]/2)-(proposals[i,1]-proposals[i,3]/2))+1) + (box_area - iw*ih)
                    overlaps = (iw*ih) / ua
                    # print(proposals[i], proposals[j], overlaps, iw, ih, ua, box_area)
                    if (overlaps > 0.5 and j > i):
                        discard = True
        if (not discard):
            nms_proposals.append(proposals[i])
    return nms_proposals

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

def readBoxesFromFile(filename):
    gt_boxes = np.empty((0,6), int)
    with open(filename) as openfileobject:
        for line in openfileobject:
            data = line.split(',')
            if(data[5] == "spiral"):
                if(data[6] == "ring\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 1, 1]
                    gt_boxes = np.vstack((gt_boxes,gt_box))
                if(data[6] == "noring\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 1, 0]
                    gt_boxes = np.vstack((gt_boxes,gt_box))
            if(data[5] == "elliptical"):
                if(data[6] == "ring\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 2, 1]
                    gt_boxes = np.vstack((gt_boxes,gt_box))
                if(data[6] == "noring\n"):
                    gt_box = [int(data[1]),int(data[2]),int(data[3]),int(data[4]), 2, 0]
                    gt_boxes = np.vstack((gt_boxes,gt_box))    
    return gt_boxes

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

@cuda.jit
def interpolation_gpu(array, grids, cellpointsfloat, cellpointsint):

    tb = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if( tb < grids.shape[0]) :
        for c in range(4):
            tx = int(cuda.threadIdx.y/3)-1
            ty = int(cuda.threadIdx.y%3)
            
            x = cellpointsfloat[tb,ty, tx, c, 0]
            y = cellpointsfloat[tb,ty, tx, c, 1]

            x1 = cellpointsint[tb,ty, tx, c, 0]    

            x2 = cellpointsint[tb,ty, tx, c, 2]

            y1 = cellpointsint[tb,ty, tx, c, 1]
            y2 = cellpointsint[tb,ty, tx, c, 3]

            l = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.z

            iXY1 = ((x2-x)/(x2-x1)*array[y1+1,x1+1,l]) + ((x-x1)/(x2-x1)*array[y1+1,x2+1,l])
            iXY2 = ((x2-x)/(x2-x1)*array[y2+1,x1+1,l]) + ((x-x1)/(x2-x1)*array[y2+1,x2+1,l])

            cuda.atomic.max(grids, (tb, ty, tx, l), ((y2-y)/(y2-y1)*iXY1) + ((y-y1)/(y2-y1)*iXY2))

def RoI_Align(array, batch_proposals, stridex, stridey, k, grids):
    cellpointsfloat = np.empty((batch_proposals.shape[0],3,3,4,2), dtype=float)
    cellpointsint = np.empty((batch_proposals.shape[0],3,3,4,4), dtype=int)
    for p in range(batch_proposals.shape[0]):
        region = batch_proposals[p,:]
        region[0] = region[0]/stridex
        region[1] = region[1]/stridey
        region[2] = region[2]/stridex
        region[3] = region[3]/stridey
        width = region[2] - region[0]
        height = region[3] - region[1]
        gwidth = width / k
        gheight = height / k

        for i in range(grids.shape[1]):
            for j in range(grids.shape[2]):
                topleftx = region[0] + j*gwidth
                toplefty = region[1] + i*gheight

                cells = np.empty((4,2), dtype=float)
                cells[0,0] = topleftx + (gwidth/2)*(0+0.5)
                cells[0,1] = toplefty + (gheight/2)*(0+0.5)
                cells[1,0] = topleftx + (gwidth/2)*(0+0.5)
                cells[1,1] = toplefty + (gheight/2)*(1+0.5)
                cells[2,0] = topleftx + (gwidth/2)*(1+0.5)
                cells[2,1] = toplefty + (gheight/2)*(0+0.5)
                cells[3,0] = topleftx + (gwidth/2)*(1+0.5)
                cells[3,1] = toplefty + (gheight/2)*(1+0.5)

                for c in range(4):
                    cellpointsfloat[p,i,j,c,0] = cells[c,0]
                    cellpointsfloat[p,i,j,c,1] = cells[c,1]
                    cellpointsint[p,i,j,c,0] = math.floor(cells[c,0])
                    cellpointsint[p,i,j,c,1] = math.floor(cells[c,1])
                    cellpointsint[p,i,j,c,2] = math.ceil(cells[c,0])
                    cellpointsint[p,i,j,c,3] = math.ceil(cells[c,1])
    threadsperblock = (6,9,16)
    blockspergrid_y = ((array.shape[2]) + (threadsperblock[2] - 1)) // threadsperblock[2]
    blockspergrid_x = ((batch_proposals.shape[0]) + (threadsperblock[0] - 1)) // threadsperblock[0]

    # start = time.time()

    d_array = cuda.to_device(array, copy=True)
    d_grids = cuda.to_device(grids, copy=True)
    interpolation_gpu[(blockspergrid_x, blockspergrid_y), threadsperblock](d_array, d_grids, cellpointsfloat, cellpointsint)

    d_grids.copy_to_host(grids)

    # print(time.time()-start)
    return grids