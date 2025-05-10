import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from typing import Tuple, List, Optional

import config
from yolo_predict import yolo_predictor
from utils import letterbox_image, load_weights

# Configure GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def initialize_model(predictor: yolo_predictor, 
                    sess: tf.Session,
                    model_path: str,
                    yolo_weights: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Initialize the YOLO model and load weights.

    Args:
        predictor: YOLO predictor object.
        model_path: Path to the model checkpoint.
        yolo_weights: Path to the YOLO weights file (optional).

    Returns:
        Tuple containing boxes, scores, and classes tensors.
    """
    input_image = tf.placeholder(tf.float32, [None, 416, 416, 3])
    input_image_shape = tf.placeholder(tf.int32, [2]) # Expect to receive a one-dimensional array of two integers:shape[h,w]
    
    # Build the prediction graph under a variable scope
    with tf.variable_scope('predict'):
        boxes, scores, classes = predictor.predict(input_image, input_image_shape)

    if yolo_weights:
        # Load YOLO weights # def load_weights(var_list, weights_file)
        load_op = load_weights(tf.global_variables(scope='predict'), yolo_weights)
        sess.run(load_op)
    else:
        # Restore model from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
    
    return boxes, scores, classes

def draw_boxes(image: Image.Image,
               boxes: List[List[float]],
               scores: List[float],
               classes: List[int],
               predictor: yolo_predictor) -> None:
    """
    Draw bounding boxes and labels on the original image.

    Args:
        boxes: List of bounding box coordinates.
        scores: List of confidence scores.
        classes: List of predicted class indices.
        predictor: YOLO predictor object.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('font/FiraMono-Medium.otf', 
                             size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')) # 3% * h + 0.5 (方便取整四舍五入) 
    thickness = (image.size[0] + image.size[1]) // 300                                  # Box thickness = (w + h)/300 (整除)
    
    # Draw in reverse order to match top-to-bottom in image
    for i, (box, score, cls_idx) in reversed(list(enumerate(zip(boxes, scores, classes)))):
        label = f"{predictor.class_names[cls_idx]} {score:.2f}"
        label_size = draw.textsize(label, font) # Calculate text size
        
        # Convert box coordinates to integers
        # top, left, bottom, right = map(lambda x: int(np.clip(x, 0, image.size[1]-1)), box)
        # box: [ymin, xmin, ymax, xmax]  
        top, left, bottom, right = box

        top = np.clip(np.floor(top + 0.5), 0, image.size[1] - 1).astype('int32')
        left = np.clip(np.floor(left + 0.5), 0, image.size[0] - 1).astype('int32')
        bottom = np.clip(np.floor(bottom + 0.5), 0, image.size[1] - 1).astype('int32')
        right = np.clip(np.floor(right + 0.5), 0, image.size[0] - 1).astype('int32')
        
        # Draw bounding box
        for t in range(thickness):
            draw.rectangle([left + t, top + t, right - t, bottom - t],  # bold inward, width=thickness,
                          outline=predictor.colors[cls_idx])
        
        # Calculate text position (above box if possible)
        text_y = top - label_size[1] if top - label_size[1] >= 0 else top + 1
        text_origin = (left, text_y)
        
        # Draw label background and text
        draw.rectangle([text_origin, (text_origin + label_size)],
                      fill=predictor.colors[cls_idx])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        
def detect(image_path: str, 
           model_path: str, 
           yolo_weights: Optional[str] = None) -> None:
    """
    Loads a YOLO model and performs detection on the given image. (Main function for detection).
    Args: model_path: Path to the trained model checkpoint.(used when yolo_weights is None)
    """
    # Preprocess the image -->feed_dict(resize_image, size)
    image = Image.open(image_path)
    resized_image = letterbox_image(image, (416, 416))
    image_data = np.array(resized_image, dtype=np.float32) / 255.0
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

    with tf.Session() as sess:
        try:
            # Initialize the predictor
            predictor = yolo_predictor(config.obj_threshold, config.nms_threshold,
                                       config.classes_path, config.anchors_path)
            
            # Initialize the model
            boxes, scores, classes = initialize_model(predictor, sess, model_path, yolo_weights)
            
            # Run prediction
            feed_dict = {
                input_image: image_data,
                input_image_shape: [image.size[1], image.size[0]]
            }   # Note: input_image_shape is provided as [height, width]
            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict=feed_dict)

            # Visualize results
            print(f'Found {len(out_boxes)} objects')
            draw_boxes(image, out_boxes, out_scores, out_classes, predictor)
            
            # Save results
            output_path = './img/result.jpg'
            image.save(output_path)
            image.show()

        except tf.errors.OpError as e:
            print(f"Model loading failed: {e}")
        except Exception as e:
            print(f"Error during detection: {e}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--image', type=str, default=config.image_file, help='Path to the input image')
    parser.add_argument('--model', type=str, default=config.model_dir, help='Path to the model checkpoint')
    parser.add_argument('--weights', type=str, default=config.yolo3_weights_path if config.pre_train_yolo3 else None,
                       help='Path to the YOLO weights file')
    args = parser.parse_args()

    # Run detection
    detect(args.image, args.model, args.weights)
