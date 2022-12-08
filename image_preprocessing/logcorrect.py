from PIL import Image
import math
import os
import numba.cuda as cuda
import numpy as np
import cropper

@cuda.jit
def maxIntensityGpu(imArray, maxIntensity):
    x, y = cuda.grid(2)
    intensity =  imArray[x,y,0] + imArray[x,y,1] + imArray[x,y,2]
    cuda.atomic.max(maxIntensity, 0, intensity)

@cuda.jit
def logcorrectGpu(imArray, logMaxIntensity):
    x, y = cuda.grid(2)
    currentIntensity = math.log(imArray[x,y,0] + imArray[x,y,1] + imArray[x,y,2])
    if(currentIntensity != 0):
        imArray[x,y,0] = int(imArray[x,y,0] * (currentIntensity / logMaxIntensity))
        imArray[x,y,1] = int(imArray[x,y,1] * (currentIntensity / logMaxIntensity))
        imArray[x,y,2] = int(imArray[x,y,2] * (currentIntensity / logMaxIntensity))
    cuda.syncthreads()

def noiseremoval(im):
    max_intensity = 0
    for x in range(im.width):
        for y in range(im.height):
            r,g,b = im.getpixel((x,y))
            if((r+g+b)>max_intensity):
                max_intensity=(r+g+b)
    max_intensity = int(max_intensity)
    for x in range(im.width):
        for y in range(im.height):
            r,g,b = im.getpixel((x,y))
            current_intensity = int(r+g+b)
            if(current_intensity != 0):
                r = int(r*math.log(current_intensity,max_intensity))
                g = int(g*math.log(current_intensity,max_intensity))
                b = int(b*math.log(current_intensity,max_intensity))
                im.putpixel((x,y),(r,g,b))

def removeandsave(imagepath, imagename):
    im = Image.open("%s/original/%s" % (imagepath, imagename))

    imArray = np.asarray(im)
    maxIntensity = np.asarray([0])

    maxIntensity[0] = 0

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(imArray.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(imArray.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    maxIntensityGpu[blockspergrid, threadsperblock](imArray, maxIntensity)

    if(maxIntensity[0] > 0):
        logMaxIntensity = math.log(maxIntensity[0])

        logcorrectGpu[blockspergrid, threadsperblock](imArray, logMaxIntensity)

        newImage = Image.fromarray(imArray)

        imglen = len(imagename)-5
        imagenamecut = imagename[0:imglen]
        objname = imagename[0:imglen-4]

        # if not (os.path.isfile("%s/corrected/DR9_%s.png" % (imagepath, imagenamecut))):
        #     newImage.save("%s/corrected/DR9_%s.png" % (imagepath, imagenamecut), format="png")

        cropped_dr7, cropped_dr9 = cropper.crop(imagepath, objname, newImage, 24, 9)

        # if not (os.path.isfile("%s/cropped/DR9_%s.png" % (imagepath, objname))):
        #     cropped_dr7.save("%s/cropped/DR7_%s.png" % (imagepath, objname), format="png")
        #     cropped_dr9.save("%s/cropped/DR9_%s.png" % (imagepath, imagenamecut), format="png")

        if (cropped_dr7.width >= 128):
            if not (os.path.isfile("%s/clean/%s_DR9.png" % (imagepath, objname))):
                cropped_dr7.save("%s/clean/%s_DR7.png" % (imagepath, objname), format="png")
                cropped_dr9.save("%s/clean/%s_DR9.png" % (imagepath, objname), format="png")
    else:
        print("Max intensity log error at image: %s" % imagename)
