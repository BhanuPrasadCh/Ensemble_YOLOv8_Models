Code Description: Weighted Fusion of Building Footprint Extraction Models

Objective:
The goal of this code is to perform ensemble learning on building footprint extraction models that have been developed for different countries using YOLOv8. The ensemble approach aims to combine predictions from multiple models using a weighted fusion method. This method aggregates bounding boxes and masks from the models, improving the accuracy of the final predictions.

Key Components:

Intersection over Union (IoU) Calculation (bb_intersection_over_union):
A function to compute the Intersection over Union (IoU) between two bounding boxes A and B.
The IoU is calculated by finding the intersection area divided by the union of both bounding boxes.
This metric is essential for determining the overlap between predicted bounding boxes and is used later to find matching boxes for the fusion process.

Weighted Mask Fusion (get_weighted_mask):
This function fuses multiple predicted masks into a single mask based on weighted scores.
Masks are weighted by their associated scores and model contributions (if applicable).
The final mask is normalized by the total confidence to ensure proper scaling.

Weighted Box Fusion (get_weighted_box):
Similar to the mask fusion, this function fuses bounding boxes by weighting them according to the confidence scores from each model.
The final bounding box is also normalized by the total confidence score.

Finding Matching Boxes (find_matching_box):
This function identifies the most similar (matching) bounding box in a list of existing boxes by comparing their IoU with a new bounding box.
If a box exceeds a specified IoU threshold, it is considered a match, and the function returns its index.

Weighted Masks Fusion (weighted_masks_fusion):
This is the main function for the ensemble process.
It takes multiple masks, bounding boxes, and scores from different models and fuses them using the above methods.
It clusters the predictions based on IoU and performs fusion for each cluster, producing ensembled masks and bounding boxes with associated scores.
The function also allows for different types of confidence weighting and thresholding strategies.

YOLOv8 Model Prediction and Fusion:
The code initializes the paths for five YOLOv8 models trained on different datasets.
It then iterates through a directory of test images, applying each model to generate predictions (bounding boxes, masks, and scores).
These predictions are collected and passed to the weighted_masks_fusion function for ensembling.
The final ensembled masks are saved as images, with the combined masks overlaid on the original images.

Parameters and Configuration:
iou_thr: The IoU threshold used to determine if two bounding boxes are considered a match.
skip_mask_thr: The score threshold to skip low-confidence predictions.
conf_type: Specifies the type of confidence weighting to use in the fusion process. Options include 'max_weight', 'model_weight', and others.
soft_weight: A parameter used in the 'soft_weight' confidence type, controlling the balance between the number of models and their scores.
thresh_type: Determines the thresholding strategy, e.g., by number of models agreeing on a prediction.
model_weights: Allows assigning different weights to the models during fusion.

Usage:
The code is intended to be run as part of a pipeline where multiple YOLOv8 models have already been trained and saved.
The user provides paths to the models and the test dataset, and the code processes each image, performing prediction, ensembling, and saving the final results.
