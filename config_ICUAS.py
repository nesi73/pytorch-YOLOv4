import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = '/'

cfg = EasyDict()

cfg.debug = True # If true, it will show the images with the detections (cv2.imshow)
cfg.times_detected = 3 # Number of times a tile with crack has to be detected to be considered as a TP else FP

# Parameters to detect tiles
cfg.ratio_tile = [0.9, 1.1] # Ratio of the tile to detect. [min, max]
cfg.size_tile = [50, 260] # Size of the tile to detect in pixels. [min, max]
cfg.iou_tiles = 0.5 # IOU to consider two tiles as the same tile (because with canny we can detect the same tile twice in the examples (bigger and smaller inside))
cfg.margin_tile = 10 # Margin to add to the tile to detect. In pixels.

# Parameters to calculate 3D points of the tile
cfg.object_points = [[0,0,0], [0,1,0], [1,1,0], [1,0,0]] # IMPORTANT: The order of the points is important. top-left, bottom-left, bottom-right, top-right.

# Parameters to consider a TP crack
cfg.radius = 1 # Radius of the circle to detect. If detection is inside the circle, it is consider same detection, else new detection. In meters.

# Parameters to model crack detection
cfg.use_cuda = True
cfg.weights_filename = os.path.join(_BASE_DIR, 'checkpoints', 'Yolov4_epoch293.pth')
cfg.data_crack = os.path.join(_BASE_DIR, 'data', 'crack.names')
cfg.threshold = 0.7
cfg.size_image = [448, 448] # Size of the image to detect. [width, height]