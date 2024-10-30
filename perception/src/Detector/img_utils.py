import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import tensorflow as tf

# ROS 2 Humble uses 'cv_bridge' for converting ROS Image messages to OpenCV images
bridge = CvBridge()

# Callback to handle image data from ROS 2 image topic
def callback_img(data, target_size):
    img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.asarray(img, dtype=np.float32)

    return np.expand_dims(img_array, axis=0), original_img

# Crops the given image over the given bounding box coordinates, and applies zero-padding
def crop_and_pad(img, corner_min, corner_max, centered=False):
    # Ensure corner values are within image dimensions
    corner_min = np.clip(corner_min, 0, img.shape[1:])
    corner_max = np.clip(corner_max, 0, img.shape[1:])

    img = img[0,:,:,:]
    if img.shape[2] > 1:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        img = np.expand_dims(img, axis=-1)

    cropped = np.zeros(img.shape, dtype=img.dtype)
    crop = img[corner_min[1]:corner_max[1], corner_min[0]:corner_max[0], :]
    if crop.size == 0:
        return cropped

    if centered:
        startW = int((img.shape[1] - crop.shape[1]) / 2)
        startH = int((img.shape[0] - crop.shape[0]) / 2)
        cropped[startH:startH+crop.shape[0], startW:startW+crop.shape[1], :] = crop
    else:
        cropped[corner_min[1]:corner_max[1], corner_min[0]:corner_max[0], :] = crop

    cropped = np.expand_dims(cropped, axis=0)
    return cropped

# Applies median filtering on predictions
def median_filter(prediction, previous_predictions, nb_frames):
    if len(previous_predictions) < nb_frames:
        return prediction
    coords = [prev_pred[i] + pred[i] for i in range(2, len(prediction)) for prev_pred in previous_predictions]
    [x.sort() for x in coords]
    filtered = [coord[int(len(coord) / 2)] for coord in coords]
    return filtered

# Visualizes predictions on the image
def visualize(img, prediction, filtered_prediction, img_size):
    colors = ['black', 'blue', 'purple', 'green', 'red']
    classes = ['Background', 'Target 1', 'Target 2', 'Candidate', 'Forward gate']
    np_array = np.uint8(img)
    img = Image.fromarray(np_array.reshape((np_array.shape[0], np_array.shape[1], np_array.shape[2])), mode="RGB")
    draw = ImageDraw.Draw(img)

    if prediction is not None:
        for box in prediction[0]:
            xmin, ymin, xmax, ymax = box[2], box[3], box[4], box[5]
            if int(box[0]) != 0:
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                textSize = draw.textsize(label)
                draw.rectangle(((xmin-2, ymin-2), (xmin + textSize[0] + 2, ymin + textSize[1])), fill=color)
                draw.text((xmin, ymin), label, fill='white')

    return np.asarray(img, dtype=np.uint8)

# Loads a model from a JSON file
def jsonToModel(json_model_path, weights_path):
    try:
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None
