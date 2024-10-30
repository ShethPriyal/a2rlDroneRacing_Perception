import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from .utils.bounding_box import convert_coordinates


class AnchorBoxes(Layer):
    """
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.
    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.
    Input shape:
        4D tensor of shape `(batch, height, width, channels)`.
    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    """

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):

        super(AnchorBoxes, self).__init__(**kwargs)

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError(
                "`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                    this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be passed, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x):
        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in self.aspect_ratios:
            if ar == 1:
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        step_height = self.img_height / feature_map_height if self.this_steps is None else self.this_steps[0]
        step_width = self.img_width / feature_map_width if self.this_steps is None else self.this_steps[1]

        offset_height = 0.5 if self.this_offsets is None else self.this_offsets[0]
        offset_width = 0.5 if self.this_offsets is None else self.this_offsets[1]

        cy = np.linspace(offset_height * step_height,
                         (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width,
                         (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                               border_pixels='half')
        elif self.coords == 'minmax':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax',
                                               border_pixels='half')

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = tf.tile(tf.constant(boxes_tensor, dtype='float32'), (tf.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.n_boxes, 8

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

