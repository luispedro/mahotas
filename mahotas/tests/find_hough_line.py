# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jasson <1917098992@qq.com>
# License: MIT (see COPYING file)
import collections
import math
import numpy as np
def find_hough_lines(im):
    '''
            line = find(img={square})
            detect lines in the image using the Hough transform
            Parameters
            ----------
            img : ndarray
                input. the binary image conatins lines(the value equals one, the backgroud value equals zero).
            Returns
            -------
             line : ndarray of line in the image
            '''
    c = LineDetectorByHough()
    return c.find(im)
class Line():
    def __init__(self,r,angle):
        self.r = r
        self.angle = angle
    def __eq__(self, other):
        if self.r == other.r and self.angle == other.angle:
            return True
        else:
            return False
    def __hash__(self):
        result = 31 + int(self.r)
        result = 31 * result + int(self.angle * 100)
        return result
class LineDetectorByHough:
    countersDICT = collections.Counter()
    def __init__(self):
        pass

    def find(self,img):
        rows, cols = img.shape

        points = 0
        for x in range(0, rows):
            for y in range(0, cols):
                if (img[x, y] == 1):
                    points = points + 1
                    for angle in np.arange(0, 2 * np.pi, 2 * np.pi / 40):
                        # print 'angel=',x, y, angel
                        angle = round(angle, 2)
                        angle = angle % round(np.pi, 2)
                        r = round(self.calcR(x, y, angle),1)
                        #print 'r,angel=', r, angle
                        self.countersDICT[Line(r, angle)] += 1
        #print "points=", points
        #print "line (r ,angle) count"
        for (k, v) in self.countersDICT.items():
            if(v > rows/2):
                return Line(k.r, k.angle)
    def calcR(self,x,y,angle):
        return x* math.cos(angle) + y * math.sin(angle)
