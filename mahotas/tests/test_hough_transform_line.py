# -*- coding: utf-8 -*-
# Copyright (C) 2008-2014, Jasson <1917098992@qq.com>
# License: MIT (see COPYING file)
import unittest
import numpy as np
import math

from LineDetectorByHough import LineDetectorByHough
from LineDetectorByHough import Line


class test_hough_transform_line(unittest.TestCase):

    def test_simpleImg_hough_detect(self):
        img = np.zeros((80, 80), bool)
        img[6,:] = 1

        c = LineDetectorByHough()
        lines = c.find(img)

        self.assertEquals(lines,Line(6,0))

    def test_counter(self):
        c = LineDetectorByHough

        line = Line(10,np.pi/4)
        c.countersDICT[line] += 1

        line = Line(10,np.pi/4)
        c.countersDICT[line] += 2

        self.assertEquals(c.countersDICT[line], 3)

    def test_line_r_calc(self):
        c = LineDetectorByHough()
        self.assertEquals(c.calcR(10,10,np.pi/4), math.sqrt(10*10 + 10*10))




