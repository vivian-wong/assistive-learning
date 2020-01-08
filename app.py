from flask import Flask, render_template, request

import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

from vivian import * # vivian.py
import mask_rcnn_max.vivian_detect_mask_rcnn as vivian_detect_mask_rcnn

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import os

import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
from torch.utils.data import DataLoader
import shutil 
from IPython.display import clear_output
from collections import defaultdict

import base64
import json
import train

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/testYOLOV3", methods=['POST'])
def testYOLOV3():
	# DETECT()
	images = request.get_data().decode('utf-8')[2:].replace('%2F','/') # one image e.g. data/GDXray/images/Castings/C0029/C0029_0030.png
	print(images)
	weights = 'weights/medAL685.pt' # to be filled in 
	
	# default, normally don't change 
	cfg = 'cfg/yolov3-spp.cfg'
	data_cfg = 'data/GDXray.data'
	img_size = 416
	conf_thres = 0.5 # 0.001 in original code 
	nms_thres = 0.5 # iou threshold for non-maximum suppression
	iou_thres = 0.5 # originally 0.5
	batch_size = 32

	# call vivian's detection function 
	# return a json of boxes info
	results = detectYOLOV3(cfg, data_cfg, weights, images, img_size, conf_thres, nms_thres) # vivian.py -> detect()
	return results

@app.route("/testMaskRCNN", methods=['POST'])
# call vivian_detect_mask_rcnn.detect()
def testMASKRCNN():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'	
    im = request.get_data().decode('utf-8')[2:].replace('%2F','/') # one image e.g. data/GDXray/images/Castings/C0029/C0029_0030.png
    print(im)
    weights = os.path.join(os.getcwd(), "mask_rcnn_max", "mask_rcnn_gdxray_0160.h5") # to be filled in 

    return vivian_detect_mask_rcnn.detect(im, weights)

@app.route("/sample", methods = ['POST', 'GET'])
def sample():
	# # Sample all and put in medAL_sampling.txt

	# # TO DO

	# # load "distances to trained " dictionary from JSON
	# data = request.get_data().decode('utf-8')
	# print(data)
	# if data == '':
	# 	print('initialize')
	# 	distances_to_trained = {}
	# else:
	# 	distances_to_trained = json.loads(data)

	# # Contexual sampling 
	# data_json = sampleNextImage(distances_to_trained)
	# return data_json

	print("Sampling")
	with open("metadata/GDXray/medAL_sampling.txt") as f:
		txt = f.read()
	return txt

@app.route("/create_labels", methods = ['POST', 'GET'])
def create_labels():
	img_size = 416

	# create a label txt file for a particular image , put it in data/GDXray/interface_generated_labels
	data = request.get_data().decode('utf-8')
	data = json.loads(data)
	s = data['backgroundImage']['src'] # e.g. http://localhost/annotatingInterface/yolov3/data/GDXray/images/Castings/C0001/C0001_0017.png
	# get label path
	label_path = 'data/GDXray/interface_generated_labels/'+'/'.join(s.split('/')[-3:]).replace('.png','.txt')
	
	# write labels 
	objects = data['objects']
	with open(label_path,"w") as file:
		for d in objects:
			if d['type'] == 'rect':
			    x1,y1, w,h = d['left'],d['top'],d['width'], d['height']
			    x = x1+w/2
			    y = y1+h/2
			    
			    x = x/img_size
			    y = y/img_size
			    w = w/img_size
			    h = h/img_size
			    
			    label = '0 {0} {1} {2} {3}\n'.format(x, y, w, h) # in "0 class x_center y_center width height"
			    file.write(label)
	return "Finished writing to "+label_path

@app.route("/train", methods=['POST','GET'])
def train_increment():
	# TRAIN() 
	# TO DO 
	return "Done training? val mAP"
