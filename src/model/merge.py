import numpy as np
import cv2
from src.model.yolo import draw_prediction
import matplotlib.pyplot as plt


def merge_labels(bb, img, filtered_points, class_ids, classes, COLORS):
    # Bounding boxes
    boxes = bb
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(img[:, :, 0])
    lab_radar_dict = {}
    # Draw Bounding Boxes
    for i in range(len(bb)):
        x1, y1, x2, y2 = bb[i]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        mask[y1:y2, x1:x2] = 255
        confidence = 0  # dummy value
        draw_prediction(img, class_ids[i], confidence, x1, y1, x2, y2, classes, COLORS)

    # Create Dictionary & Plot Points within BB
    x_prime, y_prime = filtered_points[:, 0], filtered_points[:, 1]
    bb_count = 0
    for ids in class_ids:
        label = str(classes[ids])
        label = label + str(bb_count)
        color = COLORS[ids]
        for idx in range(0, len(x_prime)):
            x = int(x_prime[idx])
            y = int(y_prime[idx])
            x1, y1, x2, y2 = bb[bb_count]
            if x1 <= x <= x2 and y1 <= y <= y2:
                lab_radar_dict[x, y] = label
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        bb_count += 1

    # Plot points that are not in BB
    for idx in range(0, len(x_prime)):
        x = int(x_prime[idx])
        y = int(y_prime[idx])
        if (x, y) not in lab_radar_dict:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    return img,lab_radar_dict
