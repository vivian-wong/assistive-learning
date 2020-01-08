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
from collections import defaultdict
import json

# input: a list of all trained image paths, a list of all available image paths 
# output: append full image path of the next image that should be annotated to new_metadata
#         a distances_to_trained dictionary storing {img path:np array of distances to all imgs in trained_imgs}
def sampleNextImage (distances_to_trained,
                     trained_image_metadata= "./metadata/GDXray/medAL_sampling.txt", 
                     all_image_metadata = "./metadata/GDXray/castings_shuffled_685.txt", 
                     new_metadata = "./metadata/GDXray/medAL_sampling.txt"):
    if trained_image_metadata != new_metadata:
        shutil.copy(trained_image_metadata, new_metadata)
        
    # parse txt files
    def read_paths(metadata):
        image_paths = []
        with open(metadata,"r") as f:
            image_paths += f.readlines()

        # Strip all the newlines
        image_paths = [p.rstrip() for p in image_paths]
        return image_paths
    trained_image_paths = read_paths(trained_image_metadata)
    all_image_paths = read_paths(all_image_metadata)
    
    # make a list of trained images as np arrays
    trained_images = []
    trained_specimens = defaultdict(int) # {specimen:count of this specimen in trained data}
    for p in trained_image_paths:
        im = cv2.imread(p).flatten() # shape = (416*416*3,)
        trained_images.append(im)
        trained_specimens[os.path.basename(os.path.split(p)[0])] += 1
    
    
    # compute distances 
    untrained_images = []
    max_avg_dist = -1
    max_avg_dist_unique_specimen = -1
    next_image_path = None
    for p in all_image_paths: 
        if p not in trained_image_paths:
            # modify dictionary so that every value is subtracted by min . that way we can pick the specimen with least images
            min_occurance = min(trained_specimens.values())
            for spec in trained_specimens:
                trained_specimens[spec] -= min_occurance
            
            im = cv2.imread(p).flatten()
            untrained_images.append(im)
            if p not in distances_to_trained:  
                distances = np.sum(np.square(im - trained_images), axis = 1) # actually is distance squared, length = # trained img
                distances_to_trained[p] = distances
            else:
                assert(len(distances_to_trained[p]) == len(trained_images)-1)
                distance_to_last_sampled = np.sum(np.square(im - trained_images[-1])) # len = 0
                distances_to_trained[p]=np.append(distances_to_trained[p],distance_to_last_sampled)
                distances = distances_to_trained[p]
            
            # find next best image
            # find the image that has a specimen that has not been chosen yet AND with max avg distance 
            specimen=os.path.basename(os.path.split(p)[0])
            avg_dist = np.mean(distances)
            
            if avg_dist > max_avg_dist_unique_specimen and (specimen not in trained_specimens or trained_specimens[specimen]==0):
                next_image_path = p 
                max_avg_dist_unique_specimen = avg_dist 
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist 
                if max_avg_dist_unique_specimen == -1:
                    next_image_path = p
    
    # write next_image_path to new_metadata
    with open(new_metadata,"a") as f: 
        f.write("\n" + next_image_path)
    
    print("Wrote " + next_image_path + " to " + new_metadata)
    print("which now has " + str(len(trained_images) + 1) + " images")

    result = {'distances':distances_to_trained, 'next_image': next_image_path}
    data_json = json.dumps(result, sort_keys=True, indent=4, separators=(',',':'))
    return data_json 

# cleaned up version of original yolov3/detect.py 
def detectYOLOV3(
        cfg,
        data_cfg,
        weights,
        images,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5):
    results = []
    # load model 
    device = torch_utils.select_device()
    model = Darknet(cfg, img_size).to(device)

    # load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    # set to eval mode
    model.to(device).eval()
    # Set Dataloader
    dataloader = LoadImages(images, img_size=img_size)
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # detect
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(img)
            detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                x1,y1,x2,y2 = xyxy
                class_name = classes[int(cls)]
                detect_result = {'class':class_name, 'x1':x1.item(),'y1':y1.item(),'x2':x2.item(),'y2':y2.item(), 'conf':conf.item(), 'cls_conf':cls_conf.item()}
                results.append(detect_result)

        print('Done. (%.3fs)' % (time.time() - t))
    print('RESULTS: ', results)
    data_json = json.dumps(results, sort_keys=True, indent=4, separators=(',',':'))
    return data_json 