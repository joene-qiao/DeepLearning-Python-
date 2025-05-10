import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def convert_coco_bbox(size, box):
    """
    Converts COCO bounding box format to normalized (x, y, w, h) relative to the image size.

    Parameters:
        size (tuple): Original image size (width, height).
        box (list): COCO bounding box [x_min, y_min, width, height].
    """
    dw, dh = 1. / size[0], 1. / size[1] # to normalize coordinate between 0 and 1
    x = (box[0] + box[2] / 2.0) * dw    # the center xy-coordinat
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h


def box_iou(boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Calculate IoU (Intersection over Union) between boxes and cluster centers.
    
    Args:
        boxes: Array of box dimensions (width, height) with shape (N, 2)
        clusters: Array of cluster center dimensions (width, height) with shape (K, 2)
        
    Returns:
        iou_matrix: IoU values between each box and each cluster (N, K)
        
    Note:
        Assumes all boxes and clusters have their top-left corners at (0,0)
    """
    box_area = boxes[:, 0] * boxes[:, 1]            # Shape: (N,)
    cluster_area = clusters[:, 0] * clusters[:, 1]  # Shape: (K,)
    
    min_width = np.minimum(boxes[:, None, 0], clusters[:, 0])  # (N, 1) and (K,) ->(N, K)
    min_height = np.minimum(boxes[:, None, 1], clusters[:, 1])
    
    intersection = min_width * min_height  # Shape: (N, K)
    union = (box_area[:, None] + cluster_area - intersection)
    return intersection / union


def avg_iou(boxes, clusters):
    """
    Calculate average maximum IoU between boxes and cluster centers.
    
    Args:
        boxes: Array of box dimensions (width, height) with shape (N, 2)
        clusters: Array of cluster center dimensions (width, height) with shape (K, 2)
        
    Returns:
        Average of the maximum IoU values between each box and its closest cluster
    """
    iou_matrix = box_iou(boxes, clusters)  # Shape: (N,K)
    return np.mean(np.max(iou_matrix, axis=1))


def kmeans(boxes, k, max_iter=25, update_fn=np.median):
    """
    Performs K-Means clustering on bounding box dimensions.
    
    Parameters:
        boxes (ndarray): Bounding box dimensions (width, height).
        k (int): Number of clusters.
        max_iter (int): Stopping criterion for iterations.
        update_fn (function): Method to update cluster centers.
    
    Returns:
        list: Final cluster centers scaled to 416 pixels.
        float: Best average IoU.
    """
    np.random.seed()
    clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]
    best_iou, best_clusters, iter_count = 0, None, 0
    
    while True:
        distances = 1. - box_iou(boxes, clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        avg_iou_score = np.mean(1. - np.min(distances, axis=1))
        
        if avg_iou_score > best_iou:
            best_iou, best_clusters = avg_iou_score, clusters.copy()
            iter_count = 0
        else:
            iter_count += 1
            if iter_count >= max_iter:
                break
        
        for i in range(k):
            clusters[i] = update_fn(boxes[nearest_clusters == i], axis=0)
    
    anchors = [[round(c[0] * 416), round(c[1] * 416)] for c in best_clusters]
    return anchors, best_iou


def load_coco_dataset(ann_file):
    """
    Loads bounding boxes from a COCO dataset annotation file.
    
    Parameters:
        ann_file (str): Path to COCO annotation file.
    
    Returns:
        ndarray: Normalized bounding box widths and heights.
    """
    coco = COCO(ann_file)
    image_ids = coco.getImgIds()
    data = []
    
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            data.append(convert_coco_bbox((img_info['width'], img_info['height']), ann['bbox'])[2:])
    
    return np.array(data)


def process(ann_file, k, max_iter=25, update_fn=np.median):
    """
    Main function to process COCO annotations and perform K-Means clustering.
    
    Parameters:
        ann_file (str): Path to COCO annotation file.
        k (int): Number of clusters.
        max_iter (int): Stopping criterion for iterations.
        update_fn (function): Method to update cluster centers.
    """
    boxes = load_coco_dataset(ann_file)
    plt.scatter(boxes[:1000, 1], boxes[:1000, 0], c='r')
    
    best_anchors, best_iou = None, 0
    for _ in range(100):
        anchors, iou = kmeans(boxes, k, max_iter, update_fn)
        if iou > best_iou:
            best_anchors, best_iou = anchors, iou
            print(f"Updated anchors: {best_anchors}, Avg IoU: {best_iou:.4f}")
    
    plt.scatter(np.array(best_anchors)[:, 1], np.array(best_anchors)[:, 0], c='b')
    plt.show()
    print(f"Final anchors: {best_anchors}, Avg IoU: {best_iou:.4f}")


if __name__ == '__main__':
    process('./annotations/instances_train2014.json', 9)
