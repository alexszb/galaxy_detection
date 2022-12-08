cimport cython
import numpy as np
cimport numpy as np


def fillWithGalaxy(
        arrayToFill,
        galaxyArray,
        padding_top,
        padding_left,
        galaxyHeight,
        galaxyWidth):
		
    arrayToFill[padding_top:padding_top+galaxyHeight, padding_left:padding_left+galaxyWidth] += galaxyArray

    return arrayToFill