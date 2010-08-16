import numpy as np
import _convex

def fill_polygon(polygon, canvas, color=1):
    '''
    fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)

    Draw a filled polygon in canvas

    Parameters
    ----------
      polygon : a list of (y,x) points
      canvas : where to draw
      color : which colour to use (default: 1)
    '''
# algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
    min_y = min(y for y,x in polygon)
    max_y = max(y for y,x in polygon)
    polygon = [(float(y),float(x)) for y,x in polygon]
    for y in xrange(min_y, max_y+1):
        nodes = [] 
        j = -1
        for i,p in enumerate(polygon):
            pj = polygon[j]
            if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                nodes.append( (p[1] + (y-p[0])/(pj[0]-p[0])*(pj[1]-p[1])) )
            j = i
        nodes.sort()
        for n,nn in zip(nodes[::2],nodes[1::2]):
            canvas[y,n:nn] = color

def convexhull(bwimg):
    '''
    hull = convexhull(bwimg)

    Compute the convex hull as a polygon
    '''
    Y,X = np.where(bwimg)
    P = list(zip(Y,X))
    if len(P) <= 3:
        return []
    return _convex.convexhull(P)

def fill_convexhull(bwimg):
    '''
    hull = fill_convexhull(bwimg)

    Compute the convex hull and return it as a binary mask
    '''

    points = convexhull(bwimg)
    canvas = np.zeros_like(bwimg)
    black = (1 if bwimg.dtype == np.bool_ else 255)
    fill_polygon(points, canvas, black)
    return canvas

