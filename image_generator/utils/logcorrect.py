import math
import os
import numba.cuda as cuda
import numpy as np
from PIL import Image

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

def log_correction(imArray):
    maxIntensity = np.asarray([0])

    maxIntensity[0] = 0

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(imArray.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(imArray.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_imArray = cuda.to_device(imArray, copy=True)
    d_maxIntensity = cuda.to_device(maxIntensity, copy=True)
    maxIntensityGpu[blockspergrid, threadsperblock](d_imArray, d_maxIntensity)
    d_maxIntensity.copy_to_host(maxIntensity)

    if(maxIntensity[0] > 0):
        logMaxIntensity = math.log(maxIntensity[0])
        logcorrectGpu[blockspergrid, threadsperblock](d_imArray, logMaxIntensity)
        d_imArray.copy_to_host(imArray)

    return imArray