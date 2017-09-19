import sys

from video.videoclass_new import VideoTestAndroid
sys.path.append("..")
from ssd.ssd import SSD300 as SSD
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
input_shape = (300,300,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('weight/weights_SSD300.hdf5')
        
vid_test = VideoTestAndroid(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
# vid_test.run('https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00002.00349.mp4')
vid_test.run(0)
