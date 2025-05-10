import json
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict

def load_weights(var_list, weights_file):
    """
    Load pre-trained Darknet53 weights into TensorFlow variables.

    Args: var_list: List of TensorFlow variables to assign weights.
          weights_file: Path to the binary weights file.

    Returns: assign_ops: List of TensorFlow assignment operations.
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)  # Skip the header (first 5 values)
        weights = np.fromfile(fp, dtype=np.float32)  

    ptr = 0          #  a pointer to keep track of the current position in the weights array.
    assign_ops = []  #  assign operations
    
    i = 0
    while i < len(var_list) - 1:
        var1, var2 = var_list[i], var_list[i + 1]
        
        if 'conv' in var1.name.split('/')[-2]: 
            if 'bn' in var2.name.split('/')[-2]:
                # Load batch normalization parameters
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                for param in [beta, gamma, mean, var]:
                    shape = param.shape.as_list()  # shape = [beta, gamma, mean, var]
                    num_params = np.prod(shape)    # num_params = beta * gamma * mean * var
                    param_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(param, param_weights))

                i += 4 
            elif 'conv' in var2.name.split('/')[-2]:  
                # Load biases for the conv layer
                bias = var2
                shape = bias.shape.as_list()
                num_params = np.prod(shape)
                bias_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(bias, bias_weights))
                i += 1

            # Load weights for the conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))  # 转置到列优先
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights))
            i += 1

    return assign_ops


def letterbox_image(image, size):
    """
    Resize image with aspect ratio preserved, padding the rest with gray.

    Args: image: Input PIL image.
          size: Target size (width, height).

    Returns: boxed_image: Resized and padded image.
    """
    image_w, image_h = image.size
    w, h = size
    scale = min(w / image_w, h / image_h)
    new_w = int(image_w * scale)
    new_h = int(image_h * scale)

    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image


def draw_box(image, bbox):
    """
    Draw bounding boxes on the image for TensorBoard visualization.

    Args:
        image: Input image tensor.
        bbox: Bounding box tensor with shape [batch, num_boxes, 5].
    """
    xmin, ymin, xmax, ymax, label = tf.split(bbox, num_or_size_splits=5, axis=2)

    height = tf.cast(tf.shape(image)[1], tf.float32)
    width = tf.cast(tf.shape(image)[2], tf.float32)

    # normalize bbox coordinates
    normalized_bbox = tf.concat([
        tf.cast(ymin, tf.float32) / height, 
        tf.cast(xmin, tf.float32) / width,
        tf.cast(ymax, tf.float32) / height, 
        tf.cast(xmax, tf.float32) / width
    ], axis=2)

    image = tf.cast(image, tf.float32) / 255.0

    new_image = tf.image.draw_bounding_boxes(image, normalized_bbox)
    tf.summary.image('input', new_image)


def voc_ap(rec, prec):
    """
    Calculate VOC-style Average Precision (AP).

    Args: rec: List of recall values.
          prec: List of precision values.

    Returns: ap: Average precision.
             mrec: Modified recall values.
             mpre: Modified precision values.
    """
    rec = np.array([0.0] + rec + [1.0])
    prec = np.array([0.0] + prec + [0.0])

    # Smooth precision curve
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # Calculate AP
    indices = np.where(rec[1:] != rec[:-1])[0] + 1
    ap = np.sum((rec[indices] - rec[indices - 1]) * prec[indices])
    return ap, rec, prec


def execute_weight_assignment(var_list, weights_file):
    assign_ops = load_weights(var_list, weights_file)
    with tf.Session() as sess:
        sess.run(assign_ops)
