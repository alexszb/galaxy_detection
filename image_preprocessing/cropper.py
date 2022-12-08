from PIL import Image
import numpy as np

def avgbrigthness(sy,sx, current_image, cols, rows):

    rsum = 0
    gsum = 0
    bsum = 0
    for i in range(rows):
        for j in range(cols):
            r,g,b = current_image.getpixel((sx+i,sy+j))
            rsum +=r
            gsum +=g
            bsum +=b

    totsum = rsum+gsum+bsum
    totpix = cols*rows
    avgbr = (totsum/totpix)
    return avgbr

def brigthnessmatrix(current_image, windowSize):
    totalWindows = int(current_image.height/windowSize)
    starty = 0

    Matrix = [[0 for x in range(totalWindows)] for y in range(totalWindows)]
    for i in range(totalWindows):
        startx = 0
        for j in range(totalWindows):
            avg = avgbrigthness(starty, startx, current_image, int(windowSize/2), int(windowSize/2))
            startx += windowSize
            Matrix[i][j] = int(avg)
        starty += windowSize
    return Matrix

def cropv2(current_image, windowSize, cutoff):
    Matrix = brigthnessmatrix(current_image, windowSize)
    tot_slices = len(Matrix[0])
    for i in range(tot_slices):
        for j in range(tot_slices):
            if(int(Matrix[i][j])>=100):
                print(int(Matrix[i][j]), end=' ')
            if(int(Matrix[i][j])>=10 and int(Matrix[i][j]) <100):
                print(int(Matrix[i][j]), end='  ')
            if(int(Matrix[i][j])<10):
                print(int(Matrix[i][j]), end='   ')
        print()
        
    for i in range(1, len(Matrix[0])-1):
        for j in range(1, len(Matrix[1])-1):
            if(Matrix[i-1][j] < cutoff and Matrix[i-1][j-1] < cutoff and Matrix[i][j-1] < cutoff and Matrix[i+1][j-1] < cutoff and Matrix[i+1][j+1] < cutoff and Matrix[i][j+1] < cutoff and Matrix[i-1][j+1] < cutoff and Matrix[i+1][j] <  cutoff):
                for row in range(windowSize):
                    for col in range(windowSize):
                        current_image.putpixel((i*windowSize+row,j*windowSize+col),(0,0,0))         
    return current_image

def crop(imagepath, imagename, current_image, windowSize, cutoff):
    Matrix = brigthnessmatrix(current_image, windowSize)
    tot_slices = len(Matrix[0])

    x = int(tot_slices/2)
    y = int(tot_slices/2)

    while(Matrix[y][x] > cutoff and x < tot_slices-1):
        x +=1
    if(Matrix[y][x] > cutoff):
        x +=1

    right = (x)*windowSize
    
    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and x > 0):
        x -=1
    left = x*windowSize

    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and y < tot_slices-1):
        y +=1
    if(Matrix[y][x] > cutoff):
        y +=1
    
    down = (y)*windowSize

    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and y > 0):
        y -=1
    top = y*windowSize


    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and x < tot_slices-1 and y > 0):
        x +=1
        y -=1
    topright = [y*windowSize, (x)*windowSize]
    if(Matrix[y][x] > cutoff):
        x +=1
    
    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and x > 0 and y > 0):
        x -=1
        y -=1
    topleft = [y*windowSize, x*windowSize]

    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and y < tot_slices-1 and x < tot_slices-1):
        y +=1
        x +=1
    if(Matrix[y][x] > cutoff):
        x +=1
        y +=1
    downright = [(y)*windowSize, (x)*windowSize]

    x = int(tot_slices/2)
    y = int(tot_slices/2)
    while(Matrix[y][x] > cutoff and x > 0 and y < tot_slices-1):
        y +=1
        x -=1
    if(Matrix[y][x] > cutoff):
        y +=1
    downleft = [(y)*windowSize, x*windowSize]     

    if(topleft[0] < top):
        top = topleft[0]
    if(topright[0] < top):
        top = topright[0]
    if(topleft[1] < left):
        left = topleft[1]
    if(downleft[1] < left):
        left = downleft[1]
    if(downleft[0] > down):
        down = downleft[0]
    if(downright[0] > down):
        down = downright[0]
    if(downright[1] > right):
        right = downright[1]
    if(topright[1] > right):
        right = topright[1]
    
    xside = max(left,current_image.width-right)
    yside = max(top,current_image.height-down)
    
    if(xside > 144):
        xside = 144
    if(yside > 144):
        yside = 144

    im_dr7 = Image.open("%s/original/%s_DR7.jpeg" % (imagepath, imagename))
    im_dr9 = Image.open("%s/original/%s_DR9.jpeg" % (imagepath, imagename))
    # current_image = current_image.crop((xside,yside,current_image.width-xside,current_image.height-yside))
    smaller_side_cut = min(xside, yside)

    dr7_cropped = im_dr7.crop((smaller_side_cut,smaller_side_cut,current_image.width-smaller_side_cut,current_image.height-smaller_side_cut))
    dr9_cropped = im_dr9.crop((smaller_side_cut,smaller_side_cut,current_image.width-smaller_side_cut,current_image.height-smaller_side_cut))
    return dr7_cropped, dr9_cropped
