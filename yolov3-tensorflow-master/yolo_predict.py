import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo

class YOLOPredictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        self.obj_threshold = obj_threshold     # Confidence threshold for object detection.
        self.nms_threshold = nms_threshold     # IoU threshold for NMS.
        self.class_names = self._load_classes(classes_file)
        self.anchors = self._load_anchors(anchors_file)
        self.colors = self._generate_colors()

    
    def _load_classes(self, path):
        """Loads class names from a file."""
        with open(os.path.expanduser(path)) as f:
            return [line.strip() for line in f.readlines()]
            # return [line.strip() for line in open(object_path)]
    
    
    def _load_anchors(self, path):
        """Loads anchor values from a file."""
        with open(os.path.expanduser(path)) as f:
            return np.array([float(x) for x in f.readline().split(',')]).reshape(-1, 2)
            # return np.loadtxt(object_path, delimiter=',').reshape(-1, 2)
    
    
    def _generate_colors(self):
        """Generates distinct colors for each class."""
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # generates a list of HSV (Hue, Saturation, Value) tuples.(with fully S & V)
        colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
        # The *x syntax unpacks the tuple x into individual arguments for the function call.
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    
    
    def _decode_predictions(self, feats, anchors, num_classes, input_shape):
        """
        Decodes YOLO's raw predictions into bounding box parameters.
        Args: 
            feats: [batch_size, grid_height, grid_width, num_anchors * (num_classes + 5)]
            anchors: [num_anchors, 2], The widths and heights of the K_means anchor boxes(self.anchor)
        Returns: 
            box_xy(center), box_wh, box_confidence, box_class_probs
        """
        grid_size = tf.shape(feats)[1:3]
        # Convert the anchor boxes to align with the shape of the feature map.
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, len(anchors), 2])
        # Reshape feature map to [b, h, w, num_anchors, num_classes + 5] to extract the prediction info for each anchor box.
        predictions = tf.reshape(feats, [-1, *grid_size, len(anchors), num_classes + 5])
        
        # Compute bounding box parameters
        # Generate a grid of cell coordinates corresponding to the feature map output from the CNN
        # To map each prediction to its location in the input image, [grid_height, grid_width, 1, 2]
        grid_X, grid_Y = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.cast(tf.stack(grid_X, grid_Y, axis=-1), tf.float32)
        grid = tf.expand_dims(grid, axis=-2)
        
        # (xy + grid) / (grid_size_xy)   [b, grid_h, grid_w, num_anchors, 2]
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # exp(dwh) * Pwh / (image_wh)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        
        return box_xy, box_wh, box_confidence, box_class_probs
    
    
    def _correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Converts box coordinates from the feature map to the original image size.
        Args:
            box-xy: [b, grid_h, grid_w, num_anchors, 2] eg: [[[[0.5,0.5], [0.3,0.4]]]]
            input_shape: [416, 416]
        Return:
            Output boxes are now in [y_min, x_min, y_max, x_max] format in original image space, ready to draw or post-process.
            [b, grid_h, grid_w, num_anchors, 4]
        """
        # 1. Swap x and y to match (y, x) order used later
        box_yx = box_xy[..., ::-1]  # (center_y, center_x) 
        box_hw = box_wh[..., ::-1]  # (height, width)

        image_shape, input_shape = map(lambda x: tf.cast(x, tf.float32), (image_shape, input_shape))
        
        # 3. Compute the scaled shape of the image after letterboxing
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        # 4. Calculate how much padding was added (offset), and how the scale differs
        scale = input_shape / new_shape
        offset = (input_shape - new_shape) / 2.0 / input_shape
        
        # 5. Remove the effect of padding and scale back up to the full input shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        
        # 6. Convert (center_x, center_y, width, height) → (y_min, x_min, y_max, x_max)
        boxes = tf.concat([box_yx - (box_hw / 2.0), box_yx + (box_hw / 2.0)], axis=-1)
        # 7. Scale them back to the original image dimensions
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        
        return boxes
    
    
    def _boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        Computes bounding box coordinates and scores.
        """
        box_xy, box_wh, box_conf, box_class_probs = self._decode_predictions(feats, anchors, num_classes, input_shape)
        boxes = self._correct_boxes(box_xy, box_wh, input_shape, image_shape)
        
        boxes = tf.reshape(boxes, [-1, 4])  # Reshape to [b*13*13*3, 4]   4:[y_min, x_min, y_max, x_max]
        box_scores = tf.reshape(box_conf * box_class_probs, [-1, num_classes]) #[b*13*13*3, num_classes]
        return boxes, box_scores
    
    
    def evaluate(self, yolo_outputs, image_shape, max_boxes=20):
        """
        Applies non-max suppression (NMS) on YOLO outputs to obtain final detections.
        Args:
            yolo_outputs (list): List of feature maps from YOLO.
            image_shape (Tensor): Original image dimensions.
            max_boxes (int): Maximum number of boxes to return.

        Returns:
            boxes_ (Tensor): Final bounding box coordinates.
            scores_ (Tensor): Final detection scores.
            classes_ (Tensor): Final class predictions.
        """
        #(N,13,13,3,85), (N,26,26,3,85), (N,52,52,3,85)
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32   # → (416,416)
        
        # zip((boxes_0, scores_0), (boxes_1, scores_1), (boxes_2, scores_2))
        # -> (boxes_0, boxes_1, boxes_2), (scores_0, scores_1, scores_2)
        # will be a tuple containing all the results from all scales.
        boxes, scores = zip(*[self._boxes_and_scores(yolo_outputs[i], self.anchors[mask], 
                                                     len(self.class_names), input_shape, image_shape) 
                              for i, mask in enumerate([[6, 7, 8], [3, 4, 5], [0, 1, 2]])])
        
        boxes = tf.concat(boxes, axis=0)    # 13x13x3 + 26x26x3 + 52x52x3 = 10647 boxes total → (10647, 4)
        scores = tf.concat(scores, axis=0)  # → (10647, 80)
        
        mask = scores >= self.obj_threshold # A boolean mask
        final_boxes, final_scores, final_classes = [], [], []
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32) # Create a constant tensor: tf.Tensor(20,shape=(),dtype=int32)
        
        # Apply NMS for Each Class:
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c]) # (10647, c)-> (10647, 4): Bounding boxes that belong to class c
            class_scores = tf.boolean_mask(scores[:, c], mask[:, c]) # (10647,)
            # return the indices of the selected bounding boxes after NMS.
            indices = tf.image.non_max_suppression(class_boxes, class_scores, max_boxes_tensor, self.nms_threshold)
            
            # Selects the bounding boxes and scores based on the indices returned by NMS.
            final_boxes.append(tf.gather(class_boxes, indices))   # [max_boxes, 4]
            final_scores.append(tf.gather(class_scores, indices)) # [max_boxes]
            final_classes.append(tf.ones_like(tf.gather(class_scores, indices), dtype=tf.int32) * c) # [max_boxes]
        
        # Flattens them into single tensors. (complete list)
        # such as:[tf.constant([2, 2, 2]), tf.constant([1, 1]), ...] ->[2, 2, 2, 1, 1..]
        final_boxes = tf.concat(final_boxes, axis=0)     # (N, 4) total_number_of_boxes
        final_scores = tf.concat(final_scores, axis=0)   # (N,)
        final_classes = tf.concat(final_classes, axis=0) # (N,)
        
        return final_boxes, final_scores, final_classes
    
    
    def predict(self, inputs, image_shape):
        """
        Runs YOLO inference on an input image.
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors, self.class_names, pre_train=False)
        outputs = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        return self.evaluate(outputs, image_shape)
