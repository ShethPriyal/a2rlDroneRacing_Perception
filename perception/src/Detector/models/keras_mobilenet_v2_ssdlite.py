from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, \
    DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, ReLU
from tensorflow.keras import backend as K
from .layers.AnchorBoxesLayer import AnchorBoxes
from .layers.DecodeDetectionsLayer import DecodeDetections
from .layers.DecodeDetectionsFastLayer import DecodeDetectionsFast
from .graphs.mobilenet_v2_ssdlite_praph import mobilenet_v2_ssdlite


def predict_block(inputs, out_channel, sym, id):
    name = 'ssd_' + sym + '{}'.format(id)
    x = DepthwiseConv2D(kernel_size=3, strides=1,
                           activation=None, use_bias=False, padding='same', name=name + '_dw_conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = ReLU(6., name=name + '_dw_relu')(x)

    x = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + 'conv2')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + 'conv2_bn')(x)
    return x


def mobilenet_v2_ssd(config, mode='training'):
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes = config['n_classes'] + 1  # Account for the background class.
    l2_reg = config['l2_regularization']  # Make the internal name shorter.
    img_height, img_width, img_channels = config['input_res'][0], config['input_res'][1], config['input_res'][2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if config['aspect_ratios_global'] is None and config['aspect_ratios_per_layer'] is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` \
            cannot both be None. At least one needs to be specified.")
    if config['aspect_ratios_per_layer']:
        if len(config['aspect_ratios_per_layer']) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(config['aspect_ratios_per_layer'])))

    scales = config['scales']
    if (config['min_scale'] is None or config['max_scale'] is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    # If no explicit list of scaling factors was passed,
    # compute the list of scaling factors from `min_scale` and `max_scale`
    else:
        scales = np.linspace(config['min_scale'], config['max_scale'], n_predictor_layers + 1)

    variances = config['variances']
    if len(variances) != 4:
        raise ValueError("4 variance values must be passed, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (config['steps'] is None)) and (len(config['steps']) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (config['offsets'] is None)) and (len(config['offsets']) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if config['aspect_ratios_per_layer']:
        aspect_ratios = config['aspect_ratios_per_layer']
    else:
        aspect_ratios = [config['aspect_ratios_global']] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if config['aspect_ratios_per_layer']:
        n_boxes = []
        for ar in config['aspect_ratios_per_layer']:
            if (1 in ar) & config['two_boxes_for_ar1']:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    # If only a global aspect ratio list was passed,
    # then the number of boxes is the same for each predictor layer
    else:
        if (1 in config['aspect_ratios_global']) & config['two_boxes_for_ar1']:
            n_boxes = len(config['aspect_ratios_global']) + 1
        else:
            n_boxes = len(config['aspect_ratios_global'])
        n_boxes = [n_boxes] * n_predictor_layers

    steps = config['steps']
    offsets = config['offsets']
    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(config['subtract_mean'])

    def input_stddev_normalization(tensor):
        return tensor / np.array(config['divide_by_stddev'])

    def input_channel_swap(tensor):
        if len(config['swap_channels']) == 3:
            return K.stack(
                [tensor[..., config['swap_channels'][0]],
                 tensor[..., config['swap_channels'][1]],
                 tensor[..., config['swap_channels'][2]]], axis=-1)
        elif len(config['swap_channels']) == 4:
            return K.stack([tensor[..., config['swap_channels'][0]],
                            tensor[..., config['swap_channels'][1]],
                            tensor[..., config['swap_channels'][2]],
                            tensor[..., config['swap_channels'][3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)

    tmp_shape = K.int_shape(x1)

    if not (config['subtract_mean'] is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (config['divide_by_stddev'] is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if config['swap_channels']:
        x1 = Lambda(
            input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    links = mobilenet_v2_ssdlite(x1)

    link1_cls = predict_block(links[0], n_boxes[0] * n_classes, 'cls', 1)
    link2_cls = predict_block(links[1], n_boxes[1] * n_classes, 'cls', 2)
    link3_cls = predict_block(links[2], n_boxes[2] * n_classes, 'cls', 3)
    link4_cls = predict_block(links[3], n_boxes[3] * n_classes, 'cls', 4)
    link5_cls = predict_block(links[4], n_boxes[4] * n_classes, 'cls', 5)
    link6_cls = predict_block(links[5], n_boxes[5] * n_classes, 'cls', 6)

    link1_box = predict_block(links[0], n_boxes[0] * 4, 'box', 1)
    link2_box = predict_block(links[1], n_boxes[1] * 4, 'box', 2)
    link3_box = predict_block(links[2], n_boxes[2] * 4, 'box', 3)
    link4_box = predict_block(links[3], n_boxes[3] * 4, 'box', 4)
    link5_box = predict_block(links[4], n_boxes[4] * 4, 'box', 5)
    link6_box = predict_block(links[5], n_boxes[5] * 4, 'box', 6)

    classes_concat = Concatenate(axis=1, name='classes_concat')(
        [Reshape((-1, n_classes), name='classes_reshape1')(link1_cls),
         Reshape((-1, n_classes), name='classes_reshape2')(link2_cls),
         Reshape((-1, n_classes), name='classes_reshape3')(link3_cls),
         Reshape((-1, n_classes), name='classes_reshape4')(link4_cls),
         Reshape((-1, n_classes), name='classes_reshape5')(link5_cls),
         Reshape((-1, n_classes), name='classes_reshape6')(link6_cls)])

    boxes_concat = Concatenate(axis=1, name='boxes_concat')(
        [Reshape((-1, 4), name='boxes_reshape1')(link1_box),
         Reshape((-1, 4), name='boxes_reshape2')(link2_box),
         Reshape((-1, 4), name='boxes_reshape3')(link3_box),
         Reshape((-1, 4), name='boxes_reshape4')(link4_box),
         Reshape((-1, 4), name='boxes_reshape5')(link5_box),
         Reshape((-1, 4), name='boxes_reshape6')(link6_box)])

    return Model(inputs=x, outputs=[classes_concat, boxes_concat])

