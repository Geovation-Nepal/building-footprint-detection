# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:19:21 2017

@author: Sanjeevan Shrestha
"""

import os
import sys
import numpy as np
from PIL import Image
import finalrelaxprecision
import finalrelaxrecall



def relax_precision_recall(pred, label, relax):
    
    prediction = np.asarray(pred, dtype=np.int32)
    label = np.asarray(label, dtype=np.int32)
    positive = np.sum(prediction==1)
    true = np.sum(label==1)
    precision_tp = finalrelaxprecision.relaxprecision(pred, label, relax)
    recall_tp = finalrelaxrecall.relaxrecall(pred, label, relax)
    
    if precision_tp > positive or recall_tp > true:
        print(positive, precision_tp, true, recall_tp)
        sys.exit('calculation is wrong.')
        
    precision = precision_tp/float(positive)
    recall = recall_tp/float(true)
    f1measure = 2*precision*recall/(precision + recall)
    IOU = precision*recall/(recall+precision-(precision*recall))
    
    return precision, recall, f1measure, IOU

