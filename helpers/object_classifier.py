from dataset.detect_traffic_light import detect_object_single, \
    load_graph, classify_color_cropped_image
from helpers.vis_util import draw_bounding_boxes_on_image_array
import tensorflow as tf
import numpy as np
import random
import string
import os
from skimage.io import imsave, imread

from utils.model_util import g_type_object_of_interest, map_2d_detector


def record_image(cv_image, ref_state, save_path):
    image_saving_frequency = 1.0
    random_val = random.random()
    if random_val < image_saving_frequency:
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])

        imsave(os.path.join(save_path, random_str + "_" + str(ref_state) + '.PNG'), cv_image)


# select only the classes of traffic light
def select_boxes(boxes, classes, scores, score_threshold=0, target_class=10):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)

    return sq_boxes[sel_id]


def select_boxes_ids(boxes, classes, scores, score_threshold=0, target_class=10):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)

    return sel_id


def rearrange_and_rescale_box_elements(box, image_np):
    """
    rearrange the box elements conventions to (xmin,xmax,ymin,ymax)


    :param box:
    :param image_np:
    :return:
    """

    im_height, im_width, _ = image_np.shape
    nbox = np.zeros_like(box)

    nbox[:, 0] = box[:, 1] * im_width
    nbox[:, 1] = box[:, 3] * im_width
    nbox[:, 2] = box[:, 0] * im_height
    nbox[:, 3] = box[:, 2] * im_height
    nbox[:, 4] = box[:, 4]
    nbox[:, 5] = box[:, 5]
    return nbox


def crop_roi_image(image_np, sel_box):
    im_height, im_width, _ = image_np.shape
    (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)
    cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image


def classify_all_boxes_in_image(image_np, boxes):
    result_index_array = np.zeros(boxes.shape[0], dtype=np.int)
    for i, box in enumerate(boxes):
        cropped_image = crop_roi_image(image_np, box)
        result_color_index, _ = classify_color_cropped_image(cropped_image)
        result_index_array[i] = result_color_index

    return result_index_array


from helpers.config_tool import get_paths

data_path, artifacts_path, _ = get_paths()

detection_path = os.path.join(artifacts_path, "annotations")

import pickle


def load_detection_boxes(detection_data_path, image_path, score_threshold=[0.4 for i in range(9)],
                         target_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         rearrange_to_pointnet_convention=True,
                         output_target_class=False):
    assert len(score_threshold) == len(target_classes)

    # print(detection_data_path)
    with open(detection_data_path, 'rb') as fp:
        detection_result = pickle.load(fp)

    boxes = detection_result['boxes']
    scores = detection_result['scores']
    classes = detection_result['classes']
    num = detection_result['num']

    # print("boxes:", boxes)
    # print("scores", scores)
    # print("classes", classes)
    # print("number of detections", num)

    all_sel_boxes = None
    sq_boxes = np.squeeze(boxes)
    for idx, target_class in enumerate(target_classes):
        ids = select_boxes_ids(boxes=boxes, classes=classes, scores=scores,
                               score_threshold=score_threshold[idx], target_class=target_class)
        sel_boxes = sq_boxes[ids]
        if rearrange_to_pointnet_convention:
            image_np = imread(image_path)
            sel_boxes = rearrange_and_rescale_box_elements(sel_boxes, image_np)
        box_scores_ids = np.empty((sel_boxes.shape[0], 6))
        box_scores_ids[:, 0:4] = sel_boxes
        box_scores_ids[:, 4] = np.squeeze(scores)[ids]
        if output_target_class:
            box_scores_ids[:, 5] = np.squeeze(classes)[ids]
        else:
            box_scores_ids[:, 5] = idx
        if all_sel_boxes is None:
            all_sel_boxes = box_scores_ids
        else:
            all_sel_boxes = np.concatenate((all_sel_boxes, box_scores_ids))
    return all_sel_boxes


class LoaderClassifier(object):
    def __init__(self):
        pass

    def detect_multi_object_from_file(self, image_path, **kwargs):
        head, tail = os.path.split(image_path)
        base, ext = os.path.splitext(tail)
        det_result_path = os.path.join(detection_path, base + ".pickle")
        if not os.path.exists(det_result_path):
            raise FileNotFoundError("the image detection file {} does not exist".format(det_result_path))
        else:
            return load_detection_boxes(det_result_path, image_path, **kwargs)


class FastClassifer(object):

    def __init__(self):
        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_object(dummy_image)

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_object(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)
        num = np.squeeze(num)

        return boxes, scores, classes, num

    def detect_and_save(self, image_path, save_file):
        image_np = imread(image_path)
        det = dict()
        det['boxes'], det['scores'], det['classes'], det['num'] = self.detect_object(image_np)
        with open(save_file, 'wb') as fp:
            pickle.dump(det, fp)


class TLClassifier(object):
    def __init__(self):

        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_object(dummy_image)
        self.traffic_light_box = None
        self.classified_index = 0

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_object(self, image_np, target_class=3):

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        # print("boxes",boxes)
        # print("classes",classes)
        # print("scores",scores)

        sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores, target_class=target_class)

        if len(sel_boxes) == 0:
            return None

        sel_box = sel_boxes[0]

        return sel_box

    def detect_multi_object_from_file(self, image_path, **kwargs):
        image_np = imread(image_path)
        return self.detect_multi_object(image_np, **kwargs)

    def detect_multi_object(self, image_np, score_threshold=[0.8, 0.8, 0.8],
                            # target_classes=[3, 1, 2], coco dataset id
                            target_classes=[1, 2, 9],  # my own model
                            rearrange_to_pointnet_convention=True,
                            output_target_class=False):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :param target_class: default [car, person,cycle]
        :param rearrange_to_pointnet_convention: True to rearrange the box coordinates to the order of (xmin,xmax,ymin,ymax)
        :return:
        """

        assert len(score_threshold) == len(target_classes)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        #        print("boxes:", boxes)
        #        print("scores", scores)
        #        print("classes", classes)
        #        print("number of detections", num)

        all_sel_boxes = None
        sq_boxes = np.squeeze(boxes)
        for idx, target_class in enumerate(target_classes):
            ids = select_boxes_ids(boxes=boxes, classes=classes, scores=scores,
                                   score_threshold=score_threshold[idx], target_class=target_class)
            sel_boxes = sq_boxes[ids]
            if rearrange_to_pointnet_convention:
                sel_boxes = rearrange_and_rescale_box_elements(sel_boxes, image_np)
            box_scores_ids = np.empty((sel_boxes.shape[0], 6))
            box_scores_ids[:, 0:4] = sel_boxes
            box_scores_ids[:, 4] = np.squeeze(scores)[ids]
            if output_target_class:
                box_scores_ids[:, 5] = np.squeeze(classes)[ids]
            else:
                box_scores_ids[:, 5] = idx
            if all_sel_boxes is None:
                all_sel_boxes = box_scores_ids
            else:
                all_sel_boxes = np.concatenate((all_sel_boxes, box_scores_ids))

        return all_sel_boxes


def draw_result(image_array, nboxes):
    image_to_be_drawn = np.copy(image_array)
    strings = [[g_type_object_of_interest[map_2d_detector[int(nboxes[i, 5])]]] for i in range(nboxes.shape[0])]
    draw_bounding_boxes_on_image_array(image_to_be_drawn, nboxes[:, 0:4], display_str_list_list=strings)

    return image_to_be_drawn
