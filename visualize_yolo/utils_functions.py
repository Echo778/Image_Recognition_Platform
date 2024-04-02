import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Coordinates of the first bounding box in [x1, y1, x2, y2] format.
        box2 (list): Coordinates of the second bounding box in [x1, y1, x2, y2] format.

    Returns:
        float: IoU value.
    """
    # Calculate coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def non_max_suppression(boxes, confidences, threshold):
    """
    Apply Non-Maximum Suppression (NMS) to remove duplicated bounding boxes.

    Args:
        boxes (list): List of bounding box coordinates in [x1, y1, x2, y2] format.
        confidences (list): List of confidence scores for each bounding box.
        threshold (float): IoU threshold for removing overlapping boxes.

    Returns:
        list: Indices of selected bounding boxes after NMS.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by confidence score in descending order
    indices = np.argsort(-np.array(confidences))

    selected_indices = []
    for i in indices:
        if i in selected_indices:
            continue

        selected_indices.append(i)

        # Calculate IoU between the current box and remaining boxes
        iou_scores = [calculate_iou(boxes[i], boxes[j]) for j in indices[i + 1:]]

        # Remove indices of boxes with IoU greater than threshold
        indices_to_remove = np.where(np.array(iou_scores) > threshold)[0] + i + 1
        selected_indices = [idx for idx in selected_indices if idx not in indices_to_remove]

    return selected_indices