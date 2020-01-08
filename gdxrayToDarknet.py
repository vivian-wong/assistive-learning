"""
data loading code for Pytorch-YOLOv3 https://github.com/eriklindernoren/PyTorch-YOLOv3
converts GDXray data to Darknet Format 

modified from https://github.com/maxkferg/metal-defect-detection/blob/master/gdxray.py
"""

import os
import math
import time
import scipy
import numpy as np

import zipfile
import urllib.request
import shutil

from PIL import Image
from skimage import io, transform 
import numpy as np

# Root directory of the project
ROOT_DIR = os.getcwd()

# Classes
BACKGROUND_CLASS = 0
CASTING_DEFECT = 1
WELDING_DEFECT = 2

DATASETS = {
    "Castings": "http://dmery.sitios.ing.uc.cl/images/GDXray/Castings.zip",
    "Welding": "http://dmery.sitios.ing.uc.cl/images/GDXray/Welds.zip"
}


def load_gdxray(dataset_dir, subset, group, auto_download=False):
    """Load a subset of the GDXray dataset.
    dataset_dir: The root directory of the GDXray dataset.
    subset: What to load (train, test)
    group: If provided, only loads images that have the given classes ("Casting","Welding")
    auto_download: Automatically download and unzip GDXray images and annotations
    """

    # Read image_ids from metadata txt files 
    castings_metadata = "metadata/GDXray/castings_{0}.txt".format(subset)
    welds_metadata = "metadata/GDXray/welds_{0}.txt".format(subset)

    if group=="Castings":
        metadata = [castings_metadata]

    if group=="Welds":
        metadata = [welds_metadata]

    if group=="All":
        metadata = [castings_metadata, welds_metadata]

    image_ids = []
    for metadata_path in metadata:
        with open(metadata_path,"r") as metadata_file:
            image_ids += metadata_file.readlines()
    # Strip all the newlines
    image_ids = [p.rstrip() for p in image_ids]

    if auto_download is True:
        create_images(dataset_dir, group) # Images are in data/GDXray/images (e.g. data/GDXray/images/Castings/C0001/C0001_0004.png)
        create_labels(dataset_dir, group, image_ids) # Organize labels into data/gdxray/labels (e.g. data/GDXray/labels/Castings/C0001/C0001_0004.txt)
        resize_416(dataset_dir, group, image_ids) 

    print("Rewriting metadata ..." )
    modify_metadata(metadata,image_ids,dataset_dir)
    print("... done rewriting")

def create_labels(dataset_dir,group,image_ids):
    """
    Organize labels into data/gdxray/labels (e.g. data/GDXray/labels/Castings/C0001/C0001_0004.txt)
    """
    label_dir = os.path.join(dataset_dir, "labels") # e.g. data/GDXray/labels
    if group=="All":
                all_group = ["Castings","Welding"]
    else:
        all_group = [group]

    for group in all_group: # e.g. group = Castings 
        url = DATASETS[group]
        
        zip_file = "{0}/{1}.zip".format(label_dir, group)
        group_dir = os.path.join(label_dir, group) # e.g. group_dir =  data/GDXray/labels/Castings
        box_map = load_boxes(dataset_dir, group)  # map["Castings/C0001/C0064_0001.png"] -> [[y1, x1, y2, x2],[y1, x1, y2, x2],...]

        # Create the label txt file (and it's directory) if it doesn't exist 
        for p in image_ids: # p = "Castings/C0001/C0001_0004.png"
            _,series,image_filename = p.split('/')
            series_dir = os.path.join(group_dir,series) # e.g. series_dir = data/GDXray/labels/Castings/C0001/
            if not os.path.exists(series_dir):
                os.makedirs(series_dir)  # e.g. series_dir = data/GDXray/labels/Castings/C0001/
            
            label_path = os.path.join(label_dir,p)
            label_path = label_path.split('.')[0] + '.txt' # e.g. label_path = data/GDXray/labels/Castings/C0001/C0001_0004.txt
            
            file = open(label_path,"w")
            print("Writing labeling to " + label_path + " ...")

            # convert groundtruth.txt to darknet format and write to label txt file
            im = Image.open(os.path.join(dataset_dir,"images",p))
            width, height = im.size
            im.close()
            bboxes = box_map[p] # [[y1, x1, y2, x2],[y1, x1, y2, x2],...]
            for bb in bboxes:
                y1, x1, y2, x2 = bb
                w = (x2 - x1)
                h = (y2 - y1)
                x = (w / 2 + x1)
                y = (h / 2 + y1)
                label = '0 {0} {1} {2} {3}\n'.format(x, y, w, h) # in "0 class x_center y_center width height"
                file.write(label)

            file.close()
            print("... done writing.")

def load_boxes(dataset_dir, group):
    """
    Create a map of bounding boxes for the group
    group: Castings or Welding

    returns:
        map["Castings/C0001/C0064_0001.png"] -> [[y1, x1, y2, x2],[y1, x1, y2, x2],...]
    """
    id_format = "{group}/{folder}/{folder}_{id:04d}.png"
    group_dir = os.path.join(dataset_dir, "images", group) # data/GDXray/images/Castings
    box_map = {}

    for root, dirs, files in os.walk(group_dir):
        for folder in dirs:
            metadata_file = os.path.join(root,folder,"ground_truth.txt") # data/GDXray/images/Castings/C0001/ground_truth.txt
            if os.path.exists(metadata_file):
                for row in np.loadtxt(metadata_file):
                    row_id = int(row[0])
                    image_id = id_format.format(group=group,folder=folder,id=row_id)
                    box = [row[3],row[1],row[4],row[2]] # [y1, x1, y2, x2]
                    box_map.setdefault(image_id,[])
                    box_map[image_id].append(box)
                # Mask R-CNN expects a numpy array of boxes
                box_map[image_id] = np.array(box_map[image_id])
    return box_map

def create_images(dataset_dir, group):
    """Download and extract the zip file from GDXray

    dataset_dir: The directory to place the dataset
    group: The group to download "Castings, Welds, Both"

    Rescale & pad images so that they are 416 by 416
    """
    image_dir = os.path.join(dataset_dir, "images")

    if group=="All":
        all_group = ["Castings","Welding"]
    else:
        all_group = [group]

    for group in all_group:
        url = DATASETS[group]

        zip_file = "{0}/{1}.zip".format(image_dir, group)
        group_dir = os.path.join(image_dir, group)

        # Make the dataset dir if it doesn't exist
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Download images if not available locally
        if not os.path.exists(group_dir):
            print("Downloading images to " + zip_file + " ...")
            with urllib.request.urlopen(url) as response, open(zip_file, 'wb') as out:
                shutil.copyfileobj(response, out)
            print("... done downloading.")
            print("Unzipping " + zip_file + "...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(image_dir)
            print("... done unzipping")

            # Clean up
            print("Removing ",zip_file)
            os.remove(zip_file)

            mac_dir = os.path.join(image_dir,"__MACOSX")
            if os.path.exists(mac_dir):
                print("Removing ",mac_dir)
                shutil.rmtree(mac_dir)

    print("Finished downloading datasets")

def resize_416 (dataset_dir, group, image_ids):
    """
    Resize and overwrite all images to images of size 416x416 while keeping aspect ratio
    Pad with zeros if not square 
    Saves a new png files in e.g. data/GDXray/images/Castings/C0001/C0001_0004.png

    Convert ground truth labels [x, y, w, h] to match resized image
    """        
    desired_size = 416

    orig_image_dir = os.path.join(dataset_dir, "images_orig") # e.g. data/GDXray/images_orig
    image_dir = os.path.join(dataset_dir, "images") # e.g. data/GDXray/images
    label_dir = os.path.join(dataset_dir, "labels") # e.g. data/GDXray/labels

    # if not os.path.exists(orig_image_dir):
    #     os.makedirs(orig_image_dir)

    for p in image_ids:
        print("Resizing image " + p + "...")
        image_file = os.path.join(image_dir, p) # e.g. data/GDXray/images/Castings/C0001/C0001_0004.png
        label_file = os.path.join(label_dir, p)
        label_file = label_file.split('.')[0] + '.txt' # e.g. label_file = data/GDXray/labels/Castings/C0001/C0001_0004.txt
        
        ######## resize image ######## 
        im = Image.open(image_file)
        old_size = im.size # old_size[0] is in (width, height) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        # create a new blank square image and paste the resized on it
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        # im.save(os.path.join(orig_image_dir, p)) # save a back up of original image 
        new_im.save(image_file) # overwrite original image 

        im.close()
        new_im.close()

        ######## resize label ######## 
        print("... resizing labels for " + p + "...")
        with open(label_file,"r") as file:
            labels = np.loadtxt(file)
            if labels.ndim <= 1: 
                labels = np.array([labels])
            labels.tolist()

        # convert for rescaling & padding, normalize (takes values 0 - 1)
        newfile = open(label_file,"w")
        for l in labels: 
            class_name, x, y, w, h = l
            min_dim = np.argmin(old_size)
            padding = [0,0]  # [left/right, top/bottom]
            padding[min_dim] = (desired_size - new_size[min_dim])/2 
            new_x = (ratio * x + padding[0]) /desired_size
            new_y = (ratio * y + padding[1]) /desired_size
            new_w = ratio * w /desired_size
            new_h = ratio * h /desired_size

            # overwrite original label file 
            label = '0 {0} {1} {2} {3}\n'.format(new_x, new_y, new_w, new_h) # in normalized "0 class x_center y_center width height"
            newfile.write(label)

        newfile.close()
        print("... done resizing.")

def modify_metadata(metadata,image_ids,dataset_dir):
    """
    Modify metadata files to darknet format (i.e. make all paths absolute paths 
    Castings/C0001/C0001_0004.png => data/GDXray/images/Castings/C0001/C0001_0004.png)
    """
    abs_img_paths = []
    for im in image_ids:
        p = os.path.join(dataset_dir,"images",im) + '\n'
        abs_img_paths += p
    for metadata_path in metadata:
        new_metadata_path = metadata_path.replace(".txt", "_new.txt")
        print(new_metadata_path)
        with open(new_metadata_path,"w") as metadata_file:
            metadata_file.writelines(abs_img_paths)


############################################################
#  Main
############################################################
if __name__ == '__main__':
    load_gdxray("data/GDXray", "test", group="Castings", auto_download = True)
    load_gdxray("data/GDXray", "train", group="Castings", auto_download = True)

