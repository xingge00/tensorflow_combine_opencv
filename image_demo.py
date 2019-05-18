import os
import sys

import cv2 as cv
import numpy as np
import tensorflow as tf

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph
PATH_TO_CKPT = os.path.join("F:/Projects/PycharmProjects/opencvtest/detection/test/frozen_inference_graph.pb")

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = os.path.join("F:/Projects/PycharmProjects/opencvtest/detection/pedestrian_train/data", "label_map.pbtxt")

NUM_CLASSES = 1
detection_grap = tf.Graph()
with detection_grap.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_nump_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


with detection_grap.as_default():
    with tf.Session(graph=detection_grap) as sess:
        image_np = cv.imread("F:/Projects/PycharmProjects/opencvtest/detection/test_images/3600.jpg")
        print(image_np.shape)
        cv.imshow('input', image_np)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_grap.get_tensor_by_name('image_tensor:0')
        boxes = detection_grap.get_tensor_by_name('detection_boxes:0')
        scores = detection_grap.get_tensor_by_name('detection_scores:0')
        classes = detection_grap.get_tensor_by_name('detection_classes:0')
        num_detections = detection_grap.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.38,
            line_thickness=1)
        cv.imshow('object detection', image_np)
        cv.imwrite('/run_result.png', image_np)
        cv.waitKey(0)
        cv.destroyAllWindows()
