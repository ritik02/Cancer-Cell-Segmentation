# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:06:14 2018

@author: Vijay Gupta
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')
import cv2 as cv
count1=[]
index1=[]
countgreen=[]
high=50
for i in range(0,high):
    name = ''
    name1=''
    if i<=9:
        name = '1_2 20x 75um 1um_z0' + str(i) + '.tif'
        name1='1_2 20x 75um 1um_z0' + str(i)
    else:
        name = '1_2 20x 75um 1um_z' + str(i) + '.tif'
        name1='1_2 20x 75um 1um_z' + str(i)
    print (name)
    img = cv.imread(name,1)
    #cv.imwrite('output_intermedite_20\\original.tif',img)
    #cv.imshow('original',img)
    #cv.imwrite('output2\\'+name1+'_original_img_'+'.tif',img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    # =============================================================================
    # clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    # lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    # l, a, b = cv.split(lab)  # split on 3 different channels
    # l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    # lab = cv.merge((l2,a,b))  # merge channels
    # opening = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    # =============================================================================
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel,iterations=3)
    #cv.imshow('opening',opening)
    #cv.imwrite('output2\\'+name1+'_first_openning_'+'.tif',opening)
    #opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel,iterations=2) 
    #cv.imshow('closing',opening)
    #cv.imwrite('output2\\'+name1+'_first_closing_'+'.tif',opening)
    # convert from LAB to BGR
    #opening = img
    
    green = np.uint8([[[0,255,0 ]]])
    hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    #print( hsv_green )
    # =============================================================================
    # hsv = cv.cvtColor(opening, cv.COLOR_BGR2HSV)
    # lower_blue = np.array([10,50,50])
    # upper_blue = np.array([240,255,255])
    kernel1 = np.ones((3,3),np.uint8)
    kernel11 = np.ones((3,3),np.uint8)
    # =============================================================================
    hsv1 = cv.cvtColor(opening, cv.COLOR_BGR2HSV)
    lower_green = np.array([40,0,0])
    upper_green = np.array([80,255,255])
    mask = cv.inRange(hsv1, lower_green, upper_green)
    res = cv.bitwise_and(opening,opening, mask= mask)
    cv.imwrite('output_intermedite_20\\greens.tif',res)
    hsv2 = cv.cvtColor(res, cv.COLOR_HSV2BGR)
    cv.imwrite('output_intermedite_20\\green.tif',hsv2)
    gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
    #cv.imwrite('output_intermedite_20\\gray.tif',gray)
    #opening1 = gray
    gray=cv.GaussianBlur(gray,(5,5),0)
    opening1=cv.morphologyEx(gray, cv.MORPH_OPEN, kernel1,iterations=3)
    #cv.imwrite('output_intermedite_20\\opening.tif',opening1)
    opening11 = cv.morphologyEx(opening1, cv.MORPH_CLOSE, kernel11,iterations=3)
    #cv.imwrite('output_intermedite_20\\closing.tif',opening11)
    #opening1 = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel1)
   
    ret, thresh = cv.threshold(opening11,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #cv.imwrite('output_intermedite_20\\binary.tif',thresh)
    #thresh = cv.bitwise_not(thresh)
    # noise removal
    
    
    #opening11 = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel11)
    #opening11 =thresh
    # sure background area
    green=0
    opening11=thresh
    for ii in range(1024):
        for jj in range(1024):
            if(opening11[ii][jj]>1):
                green=green+1
    countgreen.append(green)
    sure_bg = cv.dilate(opening11,kernel11)
    #cv.imwrite('output_intermedite_20\\background.tif',sure_bg)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening11,cv.DIST_L2,3)
    #cv.imwrite('output_intermedite_20\\dist_transform.tif',dist_transform)
    tt=0.15
    ret1, sure_fg = cv.threshold(dist_transform,tt*dist_transform.max(),255,0)
    #cv.imwrite('output_intermedite_20\\foreground.tif',sure_fg)
    #cv.imwrite('output_intermedite_20\\foreground_ret1.tif',ret1)
    #sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_CLOSE, kernel11,iterations=1)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret2, markers = cv.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #cv.imwrite('output_intermedite_20\\markers.tif',markers)
    count =0
    p={}
    for i1 in range(1024):
        for j in range(1024):
            if(markers[i1][j]>1):
                count = count+1
                p[markers[i1][j]]=1
    #print(count)
    #print(len(p.keys()))
    markers = cv.watershed(img,markers)
    img[markers == -1] = [0,255,255]
    #cv.imwrite('output_intermedite_20\\final.tif',img)
    #cv.imshow('original',img)
    #cv.imwrite('output2\\'+name1+'_background_'+'.tif',sure_bg)
    #cv.imshow('image_ell',opening)
    #cv.imshow('image mask',mask)
    #cv.imwrite('output2\\'+name1+'_image_mask_'+'.tif',mask)
    #cv.imshow('foreground',sure_fg)
    #cv.imwrite('output2\\'+name1+'_foreground_'+'.tif',sure_fg)
    #cv.imshow('image_green',res)
    #cv.imwrite('output2\\'+name1+'_image_green_'+'.tif',res)
    #cv.imshow('after opening closing',opening11)
    #cv.imwrite('output2\\'+name1+'_second_closing_'+'.tif',opening11)
    #cv.imshow('image_thresd',thresh)
    #cv.imwrite('output2\\'+name1+'_image_thresh_'+'.tif',thresh)
    #cv.imshow('watershed',img)
    #cv.imwrite('output2\\'+name1+'_after_app_watershed_'+'.tif',img)
    index1.append(i)
    count1.append(len(p.keys())-2)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

for i in range(0,high):
    print('{0:.4f}'.format(countgreen[i]/(1024*1024)))

for i in range(0,high):
    plt.bar(i,count1[i])
plt.show()