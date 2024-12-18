
from __future__ import division
import tensorflow as tf
import keras.backend as K
#from keras.engine.topology import InputSpec
#from keras.engine.topology import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer



class DecodeDetectionsFast(Layer):
    """
    A Keras layer to decode the raw SSD prediction output.
    Input shape:
        3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.
    Output shape:
        3D tensor of shape `(batch_size, top_k, 6)`.
    """

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
 
        if K.backend() != 'tensorflow':
            raise TypeError(
                "This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(
                    K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError(
                "If relative box coordinates are supposed to be converted to absolute coordinates, \
                the decoder needs the image size in order to decode the predictions, \
                but `img_height == {}` and `img_width == {}`".format(
                    img_height, img_width))

        if coords != 'centroids':
            raise ValueError("The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

        # We need these members for the config.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetectionsFast, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetectionsFast, self).build(input_shape)

    def call(self, y_pred, mask=None):
        """
        Returns:
            3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
            to always yield `top_k` predictions per batch item. The last axis contains
            the coordinates for each predicted box in the format
            `[class_id, confidence, xmin, ymin, xmax, ymax]`.
        """

        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################

        # Extract the predicted class IDs as the indices of the highest confidence values.
        class_ids = tf.expand_dims(tf.to_float(tf.argmax(y_pred[..., :-12], axis=-1)), axis=-1)
        # Extract the confidences of the maximal classes.
        confidences = tf.reduce_max(y_pred[..., :-12], axis=-1, keep_dims=True)

        # Convert anchor box offsets to image offsets.
        cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[
            ..., -8]  # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[
            ..., -7]  # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6]  # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5]  # h = exp(h_pred * variance_h) * h_anchor

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        # If the model predicts box coordinates relative to the image dimensions and they are supposed
        # to be converted back to absolute coordinates, do that.
        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax,
                                                                                                axis=-1), tf.expand_dims(
                ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        # Concatenate the one-hot class confidences and the converted box coordinates
        # to form the decoded predictions tensor.
        y_pred = tf.concat(values=[class_ids, confidences, xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. Perform confidence thresholding, non-maximum suppression, and top-k filtering.
        #####################################################################################

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):
            # Keep only the non-background boxes.
            positive_boxes = tf.not_equal(batch_item[..., 0], 0.0)
            predictions = tf.boolean_mask(tensor=batch_item,
                                          mask=positive_boxes)

            def perform_confidence_thresholding():
                # Apply confidence thresholding.
                threshold_met = predictions[:, 1] > self.tf_confidence_thresh
                return tf.boolean_mask(tensor=predictions,
                                       mask=threshold_met)

            def no_positive_boxes():
                return tf.constant(value=0.0, shape=(1, 6))

            # If there are any positive predictions, perform confidence thresholding.
            predictions_conf_thresh = tf.cond(tf.equal(tf.size(predictions), 0), no_positive_boxes,
                                              perform_confidence_thresholding)

            def perform_nms():
                scores = predictions_conf_thresh[..., 1]

                # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(predictions_conf_thresh[..., -4], axis=-1)
                ymin = tf.expand_dims(predictions_conf_thresh[..., -3], axis=-1)
                xmax = tf.expand_dims(predictions_conf_thresh[..., -2], axis=-1)
                ymax = tf.expand_dims(predictions_conf_thresh[..., -1], axis=-1)
                boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=self.tf_nms_max_output_size,
                                                              iou_threshold=self.iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=predictions_conf_thresh,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima

            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1, 6))

            # If any boxes made the threshold, perform NMS.
            predictions_nms = tf.cond(tf.equal(tf.size(predictions_conf_thresh), 0), no_confident_predictions,
                                      perform_nms)

            # Perform top-k filtering for this batch item or pad it in case there are
            # fewer than `self.top_k` boxes left at this point. Either way, produce a
            # tensor of length `self.top_k`. By the time we return the final results tensor
            # for the whole batch, all batch items must have the same number of predicted
            # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
            # predictions are left after the filtering process above, we pad the missing
            # predictions with zeros as dummy entries.
            def top_k():
                return tf.gather(params=predictions_nms,
                                 indices=tf.nn.top_k(predictions_nms[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=predictions_nms,
                                            paddings=[[0, self.tf_top_k - tf.shape(predictions_nms)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(predictions_nms)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return (batch_size, self.tf_top_k, 6)  # Last axis: (class_ID, confidence, 4 box coordinates)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetectionsFast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
