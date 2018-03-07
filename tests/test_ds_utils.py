import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
from unittest import TestCase

import numpy as np
from furnace.datasets.ds_utils import *

class DSUtilsTestCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_unique_boxes(self):
        boxes = np.array([
            [10, 10, 20, 20],
            [11, 11, 22, 22],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [11, 11, 22, 22]
        ], dtype=np.float32)
        indices = unique_boxes(boxes)
        self.assertTrue((indices == np.array([0, 1, 3])).all())

    def test_box_format(self):
        xywh_boxes = np.array([
            [1.0, 1.0, 5.0, 5.0],
            [3.0, 3.0, 5.0, 5.0]
        ])
        xyxy_boxes = np.array([
            [1.0, 1.0, 5.0, 5.0],
            [3.0, 3.0, 7.0, 7.0]
        ])
        xyxy = xywh_to_xyxy(xywh_boxes)
        self.assertTrue((xyxy == xyxy_boxes).all())

        xywh = xyxy_to_xywh(xyxy_boxes)
        self.assertTrue((xywh == xywh_boxes).all())


if __name__ == '__main__':
    unittest.main()
