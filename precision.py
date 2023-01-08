# Evaluate the perfomance of trained network by evaluating intersection over union (IOU) of the  network predcition
# and ground truth of the validation set
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input images for prediction are located
# 3) Set folder for ground truth labels in Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# 4) Set number of classes number in NUM_CLASSES
# 5) Run script
###########################################################################################################################################################
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import IOU
import sys
import Data_Reader
import BuildNetVgg16
import CheckVGG16Model
import sklearn.metrics
#......................................................................................................................................
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Label_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Test image folder
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"

#-------------------------------------------------------------------------------------------------------------------------

Batch_Size=1
NUM_CLASSES = 2 #Number of classes

#VesseClasses=["Background","Vessel"] #Classes predicted for vessel region prediction
#PhaseClasses=["BackGround","Empty Vessel region","Filled Vessel region"]#

#ExactPhaseClasses=["BackGround","Vessel","Liquid","Liquid Phase two","Suspension", "Emulsion","Foam","Solid","Gel","Powder","Granular","Bulk","Bulk Liquid","Solid Phase two","Vapor"]
################################################################################################################################################################################
def main(argv=None):
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    # .........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    # .........................Build FCN Net...............................................................................................
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class for the network
    Net.build(image, NUM_CLASSES, keep_prob)  # Create the net and load intial weights
    #    # -------------------------Data reader for validation image-----------------------------------------------------------------------------------------------------------------------------

    ValidReader = Data_Reader.Data_Reader(Image_Dir,GTLabelDir=Label_Dir, BatchSize=Batch_Size) # build reader that will be used to load images and labels from validation set

    #........................................................................................................................
    sess = tf.Session()  # Start Tensorflow session
    #--------Load trained model--------------------------------------------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:  # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else: # if
        print("ERROR NO TRAINED MODEL IN: " + ckpt.model_checkpoint_path+"See TRAIN.py for training")
        sys.exit()

     fim = 0
     precision = []
     recall = []
     f1score = []
 #===========================GO over all validation images and caclulate IOU============================================================
    while (ValidReader.itr<ValidReader.NumFiles):
        print(str(fim*100.0/ValidReader.NumFiles)+"%")
        fim+=1
        
#.........................................Run Predictin/inference on validation................................................................................
        Images,  GTLabels = ValidReader.ReadNextBatchClean()  # Read images  and ground truth annotation
        #Predict annotation using net
        PredictedLabels= sess.run(Net.Pred,feed_dict={image: Images,keep_prob: 1.0})
#............................Calculate precision for prediction...............................................................

#        print("-------------------------Precision, Recall and F1_Score----------------------------------------")
        Precision = sklearn.metrics.precision_score(PredictedLabels, GTLabels.squeeze())
        Recall = sklearn.metrics.recall_score(PredictedLabels, GTLabels.squeeze())
        f1score = sklearn.metrics.f1_score(PredictedLabels, GTLabels.squeeze())
        
        print("Precision:", Precision, "Recall:", Recall, "f1score:", f1score)
        precision.append(Precision)
        recall.append(Recall)
        f1score.append(f1score)
    print("mean_precision:", np.mean(precision), "mean_recall:", recall, "mean_f1score:", f1score)

##################################################################################################################################################
main()#Run script
print("Finished")
