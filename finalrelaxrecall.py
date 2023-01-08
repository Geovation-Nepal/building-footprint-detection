# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:26:03 2017

@author: Sanjeevan Shrestha
"""

import os
import sys
import numpy as np
from PIL import Image

def relaxrecall(original, classified, relax):
    imoriginal = Image.open(original)
    imclassified = Image.open(classified)
    Pixoriginal = list(imoriginal.getdata())
    Pixclassified = list(imclassified.getdata())

    w_lim, h_lim = imoriginal.size

    truepositive = 0
    for i in range(0, h_lim, 1):
        for j in range(0, w_lim, 1):
            prediction_val = Pixoriginal[i*w_lim+j]
        
            if prediction_val == 1:
                left_top = i - relax
                if left_top <= 0:
                    left_top = 0
                    right_top = i + relax
                if right_top > h_lim:
                    right_top = h_lim-1
            
                left_down = j - relax
                if left_down <= 0:
                    left_down = 0
            
                right_down = j + relax
                if right_down > w_lim:
                    right_down = w_lim-1
             
                sum = 0
                for ii in range(left_top, right_top, 1):
                    for jj in range(left_down, right_down,1):
                        sum+= Pixclassified[ii*w_lim+jj]
                        if sum > 0:
                            truepositive+=1
    return truepositive