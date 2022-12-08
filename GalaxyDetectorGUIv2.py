from tkinter import *
from tkinter import filedialog as fd
from traceback import format_exc
from PIL import ImageDraw, Image, ImageTk
import tensorflow as tf
import numpy as np
import keras.backend as K
import gc
import os
import pathlib
import gzip
import shutil
import sys
import dbscan_detector.dbscan_detector as dbscan

solution_dir = os.path.dirname(__file__)

def detectImage(filename):
    
    galaxyResults = []
    detected_galaxies = dbscan.dbscan(filename)
    showresults = Image.open(filename)

    drawresults = ImageDraw.Draw(showresults)
    for i in range(len(detected_galaxies)):
            a = detected_galaxies[i][0]
            if(detected_galaxies[i][2] == 1 and detected_galaxies[i][1] == "spiral"):
                drawresults.line((a[0], a[1], a[0], a[3]),fill='green', width=3)
                drawresults.line((a[2], a[1], a[2], a[3]),fill='green', width=3)
                drawresults.line((a[0], a[1], a[2], a[1]),fill='green', width=3)
                drawresults.line((a[0], a[3], a[2], a[3]),fill='green', width=3)
                galaxyResults.append([a,1,1])
                # drawresults.text((a[0]-a[2]/2, a[1]+a[3]/2+20), str(odd_to_draw[i]), fill="cyan")

            if(detected_galaxies[i][2] == 0 and detected_galaxies[i][1] == "spiral"):
                drawresults.line((a[0], a[1], a[0], a[3]),fill='blue', width=3)
                drawresults.line((a[2], a[1], a[2], a[3]),fill='blue', width=3)
                drawresults.line((a[0], a[1], a[2], a[1]),fill='blue', width=3)
                drawresults.line((a[0], a[3], a[2], a[3]),fill='blue', width=3)
                galaxyResults.append([a,1,0])


            if(detected_galaxies[i][2] == 0 and detected_galaxies[i][1] == "elliptical"):
                drawresults.line((a[0], a[1], a[0], a[3]),fill='red', width=3)
                drawresults.line((a[2], a[1], a[2], a[3]),fill='red', width=3)
                drawresults.line((a[0], a[1], a[2], a[1]),fill='red', width=3)
                drawresults.line((a[0], a[3], a[2], a[3]),fill='red', width=3)
                galaxyResults.append([a,0,0])
            
            # if detected_galaxies[i][1] == "background":
            #     drawresults.line((a[0], a[1], a[0], a[3]),fill='yellow', width=3)
            #     drawresults.line((a[2], a[1], a[2], a[3]),fill='yellow', width=3)
            #     drawresults.line((a[0], a[1], a[2], a[1]),fill='yellow', width=3)
            #     drawresults.line((a[0], a[3], a[2], a[3]),fill='yellow', width=3)   

            # drawresults.text((a[0]-a[2]/2-100, a[1]-a[3]/2-20), str(classes_to_draw[i]), fill="yellow")

    # showresults.show()
    showresults.save("%s/detector_gui/results/%s_DBSCAN.jpg" % (solution_dir, filename.split("/")[-1].split(".")[0]))
    return showresults, galaxyResults

root = Tk()
root.title("Galaxy Detector")

img = ImageTk.PhotoImage(Image.open("%s/detector_gui/start_im.png" % solution_dir))
panel = Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

global skyimagename
global galaxyResults

def saveGalaxies():
    global skyimagename
    # dbscan.show_gt_boxes(skyimagename)
    for i in range(len(galaxyResults)):
        g = galaxyResults[i][0]
        loadedImage = Image.open(skyimagename)
        left = g[0]
        top = g[1]
        right = g[2]
        bottom = g[3]
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
            if galaxyResults[i][1] == 1:
                if galaxyResults[i][2] == 1:
                   crop.save("%s/detector_gui/results/ringed_spiral/%s_%s.jpg" % (solution_dir, imageNumber, i))
                else:
                   crop.save("%s/detector_gui/results/non_ringed_spiral/%s_%s.jpg" % (solution_dir, imageNumber, i))
            if galaxyResults[i][1] == 0:
                   crop.save("%s/detector_gui/results/elliptical/%s_%s.jpg" % (solution_dir, imageNumber, i))
def callback():
    global skyimagename
    skyimagename = fd.askopenfilename()
    loadedImage = Image.open(skyimagename)
    loadedImage = loadedImage.resize((1323,907))
    img2 = ImageTk.PhotoImage(loadedImage)
    panel.configure(image=img2)
    panel.image = img2
    detect.config(state="normal")
    save.config(state="disabled")

def detection():
    global galaxyResults
    results, galaxyResults = detectImage(skyimagename)
    results = results.resize((1323,907))
    img2 = ImageTk.PhotoImage(results)
    panel.configure(image=img2)
    panel.image = img2
    save.config(state="normal")
    gc.collect()

button = Button(root,
                text="Open",
                command=lambda: callback())
button.pack(side=RIGHT)

detect = Button(root,
                text="Detect",
                command=lambda: detection(),
                state = "disabled")
detect.pack(side=RIGHT)

save = Button(root,
              text="Save",
              command=lambda: saveGalaxies(),
              state = "disabled")
save.pack(side=RIGHT)

root.mainloop()