#!/usr/bin/env python3

from .models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from sensor_msgs.msg import CompressedImage, Image
from .models.losses.keras_ssd_loss import SSDLoss
from tensorflow.keras.models import load_model
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np
import yaml
from .img_utils import *

class Detector(object):
    def __init__(self, config_path, weights_path, prediction_size=(300, 225), filter_size=30):
        # Load the configuration from YAML file
        print("Init Detector!~~~~~~~~~~~~~~~~~~~~~")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.bridge = CvBridge()
        self.filter_size = filter_size
        self.original_size = (640, 480)

        # Initialize the SSD MobileNet model for inference mode
        self.model = mobilenet_v2_ssd(config, mode='inference')

        # Load the model weights
        self.model.load_weights(weights_path, by_name=True)
        print("Loaded weights from {}".format(weights_path))

        # Compile the model with SSD loss
        ssdloss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(loss=ssdloss.compute_loss, optimizer='adam')

        self.prediction_size = prediction_size
        self.previous_predictions = []

        # Optional: Configure TensorFlow GPU memory management
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)

    def predict(self, image):
        colors = ['black', 'blue', 'purple', 'green', 'red']
        classes = ['Background', 'Target 1', 'Target 2', 'Candidate', 'Backward gate']

        # Preprocess the image using your custom function
        np_image, cv_image = callback_img(image, self.prediction_size)

        # Predict bounding boxes and classes
        y_pred = self.model.predict(np_image)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold]
                         for k in range(y_pred.shape[0])]

        prediction = None
        for box in y_pred_thresh[0]:
            if int(box[0]) == 1:  # Class 1 is the 'Target 1'
                prediction = box
                break

        pred_filtered = prediction

        # Cropping the image around the prediction (if any)
        if prediction is not None:
            min_corner = (int(prediction[2]), int(prediction[3]))
            max_corner = (int(prediction[4]), int(prediction[5]))
            cropped = crop_and_pad(np_image, min_corner, max_corner, centered=False)
        else:
            cropped = None

        return cropped, prediction, pred_filtered
