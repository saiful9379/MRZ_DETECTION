import math
import cv2
import numpy as np
import os
from PIL import Image

import cv2
import numpy as np

import cv2
import numpy as np

def augment_data(image, bbox, angle):
    # Rotate the image
    rotated_image = rotate_image(image, angle)

    # Rotate the bounding box coordinates
    rotated_bbox = rotate_bbox(bbox, angle, image.shape[:2])

    return rotated_image, rotated_bbox

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_bbox(bbox, angle, image_size):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Get image center coordinates
    center_x = image_size[1] / 2
    center_y = image_size[0] / 2

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Apply rotation to the bounding box coordinates
    rotated_bbox = np.array([
        rotation_matrix[0][0] * bbox[0] + rotation_matrix[0][1] * bbox[1] + rotation_matrix[0][2],
        rotation_matrix[1][0] * bbox[0] + rotation_matrix[1][1] * bbox[1] + rotation_matrix[1][2],
        rotation_matrix[0][0] * bbox[2] + rotation_matrix[0][1] * bbox[3] + rotation_matrix[0][2],
        rotation_matrix[1][0] * bbox[2] + rotation_matrix[1][1] * bbox[3] + rotation_matrix[1][2],
    ])

    return rotated_bbox



if __name__ == "__main__":
    # box_coordinates = [59, 655, 587, 734]

    image_path = "./data/training/03-02-2023-12-14-081977_phase_1_phase_1.png"
    # image = Image.open(image_path)


    # angle = 45
    # image_width = 100
    # image_height = 100

    # r_image = rotate_image(image, angle)

    # nr_image = np.asarray(r_image)

    # r_width, r_height = r_image.size

    # rotated_box = rotate_box_coordinates(box_coordinates, angle, r_width, r_height)
    # print(rotated_box)
    # r_box = [round(i) for i in rotated_box]

    # cv2.rectangle(nr_image, pt1=(r_box[0], r_box[1]),pt2=(r_box[2], r_box[3]),color= (0,0,255),thickness=2)
    # cv2.imwrite("test.jpg", nr_image)

    # augmenter = RotateAndShift(angle_range=(-10, 10), shift_range=(-0.1, 0.1))
    # image = cv2.imread(image_path)
    # boxes = np.array([[59, 655, 587, 734], [59, 655, 587, 734]])
    # image_aug, boxes_aug = augmenter(image, boxes)

    # print(boxes_aug)
    # Example usage
    image = cv2.imread(image_path)
    bbox = [59, 655, 587, 734]
    angle = 45

    # Perform data augmentation
    augmented_image, augmented_bbox = augment_data(image, bbox, angle)

    # Display the augmented image with the rotated bounding box
    cv2.rectangle(augmented_image, (int(augmented_bbox[0]), int(augmented_bbox[1])),
                (int(augmented_bbox[2]), int(augmented_bbox[3])), (0, 255, 0), 2)
    cv2.imshow('Augmented Image', augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()