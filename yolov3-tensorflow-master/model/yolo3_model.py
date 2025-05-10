import numpy as np
import tensorflow as tf
import os


class YOLO:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        Initialize YOLO model parameters.
        Args:
            norm_epsilon: A small value added to the variance to avoid division by zero.
            norm_decay: (衰减率)Decay rate for moving average during batch normalization.
            pre_train: Whether to use pre-trained weights for Darknet53.
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors = self._load_txt(anchors_path, dtype=float)
        self.classes = self._load_txt(classes_path)
        self.pre_train = pre_train

    def _load_txt(self, file_path, dtype=str):
        """ Load and process text file data. """
        object_path = os.path.expanduser(file_path)
        if dtype == float:
            return np.loadtxt(object_path, delimiter=',').reshape(-1, 2) # Load anchor boxes
        else:
            return [line.strip() for line in open(object_path)]          # Load class names
    
    
    def _batch_norm(self, x, name, training):
        """ Batch Normalization with Leaky ReLU activation.(l2 正则化) """
        x = tf.layers.batch_normalization(x, 
                                          momentum=self.norm_decay, epsilon=self.norm_epsilon, 
                                          training=training, name=name)
        return tf.nn.leaky_relu(x, alpha=0.1)
    
    def _conv2d(self, x, filters, kernel, name, strides=1, use_bias=False):
        """ 2D Convolution with optional padding. 
        Glorot (uniform distribution)均匀分布初始化器，也称为 Xavier 均匀分布初始化器。
        """
        padding = 'SAME' if strides == 1 else 'VALID'
        return tf.layers.conv2d(x, filters, kernel, strides=[strides, strides], padding=padding,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                use_bias=use_bias, name=name)
      
    def _residual_block(self, x, filters, blocks, index, training):
        """ Darknet residual block with multiple convolutional layers. 
            conv 3x3(shortcut) -> 1x1(half filters)-> 3x3 + shortcut"""
        # Pad the input tensor with zeros(h,w)上边界和左边界各填充1个像素 保持边缘的特征更好地通过卷积层
        # half the height and width of the input feature map. 
        x = self._conv2d(tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]]), filters, 3, f"conv_{index}", strides=2)
        x = self._batch_norm(x, f"bn_{index}", training)
        index += 1
        
        for _ in range(blocks):
            shortcut = x
            x = self._batch_norm(self._conv2d(x, filters // 2, 1, f"conv_{index}"), f"bn_{index}", training)
            index += 1
            x = self._batch_norm(self._conv2d(x, filters, 3, f"conv_{index}"), f"bn_{index}", training)
            index += 1
            x += shortcut
        return x, index

    def _darknet53(self, x, index, training):
        """ Builds Darknet-53 feature extractor. 
        Returns:  Three intermediate feature maps and updated conv_index."""
        x = self._batch_norm(self._conv2d(x, 32, 3, f"conv_{index}"), f"bn_{index}", training)
        index += 1
        
        x, index = self._residual_block(x, 64, 1, index, training)
        x, index = self._residual_block(x, 128, 2, index, training)
        route1, index = self._residual_block(x, 256, 8, index, training)
        route2, index = self._residual_block(route1, 512, 8, index, training)
        route3, index = self._residual_block(route2, 1024, 4, index, training)
        
        return route1, route2, route3, index

    def _yolo_block(self, x, filters, out_filters, index, training):
        """ YOLO detection block with multiple convolutional layers(for feature extraction). """
        # route(Conv2D Block 5L): Intermediate feature map
        for _ in range(2):
            x = self._batch_norm(self._conv2d(x, filters, 1, f"conv_{index}"), f"bn_{index}", training)
            index += 1
            x = self._batch_norm(self._conv2d(x, filters * 2, 3, f"conv_{index}"), f"bn_{index}", training)
            index += 1
        x = self._batch_norm(self._conv2d(x, filters, 1, f"conv_{index}"), f"bn_{index}", training)
        index += 1
        route = x
        
        # x(conv3x3 + conv1x1): output feature map
        x = self._batch_norm(self._conv2d(x, filters * 2, 3, f"conv_{index}"), f"bn_{index}", training)
        index += 1
        x = self._conv2d(x, out_filters, 1, f"conv_{index}", use_bias=True)
        index += 1
        
        return route, x, index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        """ Builds the YOLO model using Darknet-53 as a backbone. 
         Returns:
            A list of three feature maps for detection at different scales."""
        index = 1
        route1, route2, route3, index = self._darknet53(inputs, index, training)
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        
        with tf.variable_scope('yolo'):
            #--------------------------------------#
            # First detection scale (13x13)
            # route3_mid = 13,13,512，out1 = 13,13,255(3x(80+5))
            route3_mid, out1, index = self._yolo_block(route3, 512, num_anchors * (num_classes + 5), index, training)
            
            #--------------------------------------#
            # Second detection scale (26x26)
            # step1:(conv2d 1x1 + upsampling(route3_mid)) upsample_0 = 26,26,256
            route3_mid_2 = self._batch_norm(self._conv2d(route3_mid, 256, 1, f"conv_{index}"), f"bn_{index}", training)
            index += 1
            upsample_0 = tf.image.resize_nearest_neighbor(route3_mid_2, 
                                                          size = [2 * tf.shape(route3_mid_2)[1], 2*tf.shape(route3_mid_2)[2]], 
                                                          name="upSample_0")
            # step2: concat(up,r2) = 26,26,768
            concat_0 = tf.concat([upsample_0, route2], axis=-1, name="concat_0")
            # step3: _yolo_block: route2_mid = 26,26,256，out2 = 26,26,255
            route2_mid, out2, index = self._yolo_block(concat_0, 256, num_anchors * (num_classes + 5), index, training)
            
            #--------------------------------------#
            # Third detection scale (52x52)
            # step1:(conv2d 1x1 + upsampling(route2_mid)) upsample_2 = 52,52,128
            route2_mid_2 = self._batch_norm(self._conv2d(route2_mid, 128, 1, f"conv_{index}"), f"bn_{index}", training)
            index += 1
            upsample_1 = tf.image.resize_nearest_neighbor(route2_mid_2, 
                                                          size = [2 * tf.shape(route2_mid_2)[1], 2*tf.shape(route2_mid_2)[2]], 
                                                          name="upSample_1")
            # step2: concat(up,r1) = 52,52,384
            concat_1 = tf.concat([upsample_1, route1], axis=-1, name="concat_1")
            # step3: _yolo_block: route1_mid = 52,52,128，out3 = 52,52,255
            route1_mid, out3, index = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), index, training)
            
        return out1, out2, out3
'''--------------------------------------#
Each grid point has three prior boxes, 
3x(80+4+1) : x_offset, y_offset, w, h, confidence
-->reshape
(N,13,13,3,85)
(N,26,26,3,85)
(N,52,52,3,85)
#--------------------------------------'''
