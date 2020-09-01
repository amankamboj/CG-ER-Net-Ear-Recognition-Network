"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import copy
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import warnings

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ear"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

	def load_balloon(self, dataset_dir, subset):
		"""Load a subset of the Balloon dataset.
		dataset_dir: Root directory of the dataset.
		subset: Subset to load: train or val
		"""
		# Add classes. We have only one class to add.
		self.add_class("balloon", 1, "ear")
		self.add_class("balloon", 2, "face")
		self.add_class("balloon", 3, "pan_ear")
		# Train or validation dataset?
		assert subset in ["train", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)

		# Load annotations
		# VGG Image Annotator saves each image in the form:
		# { 'filename': '28503151_5b5b7ec140_b.jpg',
		#   'regions': {
		#       '0': {
		#           'region_attributes': {},
		#           'shape_attributes': {
		#               'all_points_x': [...],
		#               'all_points_y': [...],
		#               'name': 'polygon'}},
		#       ... more regions ...
		#   },
		#   'size': 100202
		# }
		# We mostly care about the x and y coordinates of each region
		annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

		#print(len(annotations['images']))
		#print(len(annotations['annotations']))
		# The VIA tool saves images in the JSON even if they don't have any

		# Add images
		for annot in annotations['annotations']:
			i=0
			if i>10:
				print(i)
			for a in annotations['images']:
				if annot['image_id'] == a['id']:
				    i=i+1
				    # Get the x, y coordinaets of points of the polygons that make up
				    # the outline of each object instance. There are stores in the
				    # shape_attributes (see json format above)
				    xs = []
				    ys = []
				    #print("Hellooooooo")
				    #print(annot['segmentation'])
				    if len(annot['segmentation'])==1:
				    	for i in range(len(annot['segmentation'][0])):
				        	if i % 2 == 0:
				        	    xs.append(annot['segmentation'][0][i])
				        	else:
				        	    ys.append(annot['segmentation'][0][i])
				    if len(annot['segmentation'])>1:
				    	for i in range(len(annot['segmentation'])):
				        	if i % 2 == 0:
				        	    xs.append(annot['segmentation'][i])
				        	else:
				        	    ys.append(annot['segmentation'][i])
				        	    
				    #print(xs)
				    polygons = [{'all_points_x':xs, 'all_points_y':ys, 'name':'polygon'}]
				    # load_mask() needs the image size to convert polygons to masks.
				    # Unfortunately, VIA doesn't include it in JSON, so we must read
				    # the image. This is only managable since the dataset is tiny.
				    image_path = os.path.join(dataset_dir, a['file_name'])
				    height, width = annot['height'], annot['width']
				    self.add_image(
				        "balloon",
				        image_id=a['file_name'],  # use file name as a unique image id
				        path=image_path,
				        width=width, height=height,
				        polygons=polygons)

	def load_mask(self, image_id):
		#"""Generate instance masks for an image.
		#Returns:
		#masks: A bool array of shape [height, width, instance count] with
		#	one mask per instance.
		#class_ids: a 1D array of class IDs of the instance masks.
		#"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "balloon":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
						dtype=np.uint8)
		#print("ok")
		#print(len(info["polygons"]))
		try:
			for i, p in enumerate(info["polygons"]):
				# Get indexes of pixels inside the polygon and set them to 1
				rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
				mask[rr, cc, i] = 1
		except:
			print("error")
		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "balloon":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')

def my_checq(scores):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!scores!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(scores)
    return 1
    
def color_splash(image, mask, image_path='',scores=[],class_ids=[]):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = copy.deepcopy(image)
    image = np.zeros(gray.shape)
    #print("Hello")
    #print(scores)
    for row in image:
        for col in row:
            col[0] = 255
    # We're treating all instances as one, so collapse the mask into one layer
    if mask.shape[2] == 0:
        print('no mask detected')
        gray = gray.astype(int)
        return gray
    mask = (np.sum(mask, -1, keepdims=True) >= 1)

    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray

    mask_overlay = cv2.addWeighted(gray, 0.7, splash, 0.3, 0)
    return mask_overlay


def detect_and_color_splash(model, image_path=None, video_path=None, folder_path=None, save_path=None):
    assert image_path or video_path or folder_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        # with open(args.image, 'r') as f:
        #     images_path = f.read()
        #images_path = images_path.split('\n')
        #for image_path in images_path:
        print("Hello")
        image_path = args.image
        image=cv2.imread(image_path)
        cv2.imwrite('file_name.png', image)
        img = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Detect objects
        #print(model.detect([image], verbose=0))
        r = model.detect([image], verbose=0)[0]
        
        # Color splash
        splash = color_splash(img, r['masks'], image_path,r['scores'],r['class_ids'])
        rt="string"
        a=my_checq(rt)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # file_name = os.path.join('test/images', image_path.split('/')[-1])
        cv2.imwrite(file_name, splash)
        print("Saved to ", file_name)
    elif video_path:
        #import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print("Saved to ", file_name)
    elif folder_path:
        assert save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        idx = 0
        files=os.listdir(folder_path)
        already_done = os.listdir(save_path)
        for file in files:
            if file in already_done:
                continue
            idx += 1
            print(file)
            print(idx)
            image_path = os.path.join(folder_path, file)
            image=cv2.imread(image_path)
            if image is None:
                continue
            img = copy.deepcopy(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            # Color splash
            splash = color_splash(img, r['masks'], image_path)
            # Save output
            #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            # file_name = os.path.join('test/images', image_path.split('/')[-1])
            cv2.imwrite(os.path.join(save_path, file), splash)
            print("Saved to ", os.path.join(save_path, file))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--folder', required=False,
                    metavar="path to folder input",
                    help='folder to apply the color splash effect on')
    parser.add_argument('--save', required=False,
                    metavar="path or save images",
                    help='folder to save the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video or args.folder,\
               "Provide --image or --video to apply color splash"

    #print("Weights: ", args.weights)
    #print("Dataset: ", args.dataset)
    #print("Logs: ", args.logs)
    warnings.filterwarnings('ignore')

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    print("start")
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video, folder_path=args.folder, save_path=args.save)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
