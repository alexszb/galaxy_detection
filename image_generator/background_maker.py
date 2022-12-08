import pathlib
import numpy as np
import os
from PIL import Image
def getAnchorBoxes():
    anchorBoxes = np.vstack([
            [0,0,64,64],
        ])
    return anchorBoxes

def getAllAnchors(width, height, stridex, stridey):
    anchors = getAnchorBoxes()
    # anchors = [[0,0,stridex,stridex]]
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

def cutter(solution_dir, anchors, file_name):

    im = Image.open("%s/images/backgrounds/original/%s" % (solution_dir, file_name))
    im_arr = np.asarray(im)

    for i in range(len(anchors)):
        if(anchors[i][0] > 64 and anchors[i][0] < 1920 and anchors[i][1] > 64 and anchors[i][1] < 1297):
            array_cut = im_arr[int(anchors[i][1]-anchors[i][3]):int(anchors[i][1]+anchors[i][3]), int(anchors[i][0]-anchors[i][2]):int(anchors[i][0]+anchors[i][2])]
            im_cut = Image.fromarray(array_cut)
            im_cut.save("%s/images/backgrounds/clean/%s_%d.jpg" % (solution_dir,file_name[:len(file_name)-4], i))

def generate_backgrounds():
    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent
    anchors = getAllAnchors(1984,1361,64,64)

    bg_files = os.listdir("%s/images/backgrounds/original/" % solution_dir)
    for file_name in bg_files:
        cutter(solution_dir, anchors, file_name)