import os
from src.data.dataset import Scene, CameraRadarProps
from src.radar_to_image import single_radar_to_img, multi_radar_to_img
from src.radar_to_camera import single_radar_to_camera, multi_radar_to_camera
from src.camera_to_image import single_camera_to_img, multi_camera_to_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from src.model.yolo import run_yolo
from DBscan.run_dbscan import run_cluster, run_cluster_2, run_cluster_3
from src.utils import filter_radar, filter_radar_baseline
from src.model.merge import merge_labels
from src.model.yolo import draw_prediction
import numpy as np
from PIL import Image
from matplotlib import cm

# Image data
LEFT_CAM = '21248038'
MID_CAM = '20438665'
RIGHT_CAM = '21248039'

# Radar pointcloud data
LEFT_RADAR = 'ZRVE1002'
MID_RADAR = 'ZRVC2001'
RIGHT_RADAR = 'ZRVE1001'

# Data and model weights location
filepath = 'data/00049.npz'
sensor_prop_path = 'data/extrinsics_intrinsics.npz'
yolo_cfg_path = 'yolo_setup/yolov3.cfg'
yolo_class_path = 'yolo_setup/yolov3.txt'
yolo_weights_path = 'yolo_setup/yolov3.weights'
labeled_image = 'data/00118.npz_object-detection.jpg'

# Check if the YOLO files exist
if not os.path.exists(yolo_cfg_path):
    raise FileNotFoundError(f"YOLO configuration file not found: {yolo_cfg_path}")
if not os.path.exists(yolo_class_path):
    raise FileNotFoundError(f"YOLO class file not found: {yolo_class_path}")
if not os.path.exists(yolo_weights_path):
    raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights_path}")

if __name__ == '__main__':
    data = Scene(filepath)
    sensor_props = CameraRadarProps(sensor_prop_path)
    single_mode = False
    radar_data = data.get_radar_data()
    camera_data = data.get_camera_data()
    cam = LEFT_CAM
    rad = LEFT_RADAR

    # Single Radar/Image object detection
    if single_mode:
        img = camera_data[cam]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run Image Labeling
        lab_img, bb, COLORS, class_ids, classes = run_yolo(camera_data[RIGHT_CAM], yolo_cfg_path, yolo_class_path, yolo_weights_path)
        # Map Radar PC in Camera Space
        radar_camera_space = single_radar_to_camera(radar_data[rad], sensor_props.radar_extrinsic[rad], sensor_props.camera_extrinsics[cam])
        # Run DbScan
        points, unique_labels, labels, cluster_dict = run_cluster_3(radar_camera_space)
        # Map Results Radar Points in Camera Space to Image Space
        radar_ImgSpace = single_camera_to_img(radar_camera_space, sensor_props.camera_intrinsics[cam])
        # Filter
        filtered_indices, filtered_points, filtered_labels = filter_radar(radar_ImgSpace, labels)
        # Merge
        img_lab, lab_radar_dict = merge_labels(bb, img, filtered_points, class_ids, classes, COLORS, unique_labels)
    else:
        im1 = camera_data[LEFT_CAM]
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = camera_data[MID_CAM]
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        im3 = camera_data[RIGHT_CAM]
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
        radar_all = np.vstack([radar_data[LEFT_RADAR], radar_data[MID_RADAR], radar_data[RIGHT_RADAR]])

        # Run Image Labeling
        lab_img, bb, COLORS, class_ids, classes = run_yolo(im2, yolo_cfg_path, yolo_class_path, yolo_weights_path)
        # Map Radar PC in Camera Space
        radar_camera_space_dict = multi_radar_to_camera(radar_data, sensor_props.radar_extrinsic, sensor_props.camera_extrinsics[MID_CAM], sensor_props.camera_intrinsics[MID_CAM])

        # Concatenate radar points from all sensors into a single 2D array
        radar_camera_space = np.concatenate(list(radar_camera_space_dict.values()))

        # Run DbScan
        points, unique_labels, labels, cluster_dict = run_cluster_3(radar_camera_space)
        # Map Results Radar Points in Camera Space to Image Space
        radar_ImgSpace_dict = multi_camera_to_img(radar_camera_space_dict, sensor_props.camera_intrinsics[MID_CAM])
        # Flatten the filtered radar image space points
        radar_ImgSpace_filtered_flattened = np.concatenate([filter_radar_baseline(val)[1] for val in radar_ImgSpace_dict.values()])
        # Merge
        img_lab, lab_radar_dict = merge_labels(bb, im2, radar_ImgSpace_filtered_flattened, class_ids, classes, COLORS)

    # Show Radar/Image scene
    cv2.imshow('Image with highlighted points', img_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    multi_radar_to_img(radar_data, sensor_props.radar_extrinsic, sensor_props.camera_extrinsics[cam], sensor_props.camera_intrinsics[cam], image=img)
