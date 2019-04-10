
# MODEL, VIEVER, CONTROLER

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc


# Sierpinski triangle
def generate_sierpinski(max_iter):
    w = [[0,0],[0.5,1],[1,0]]
    p = [0,0]
    points = np.zeros((max_iter,2))
    for i in range(max_iter):
        k = np.random.randint(0,3)
        p[0] = 0.5*(p[0]+w[k][0])
        p[1] = 0.5 * (p[1]+w[k][1])
        points[i] = p
    return points


# sierpinski jpg
def data_to_img(data,width,height):
    img = np.zeros((height, width,3), dtype = np.uint8)
    boundary = get_boundary(data)
    # boundary = [(xmin,ymin),(xmax,ymax)]
    x_min,y_min = boundary[0]
    x_max, y_max = boundary[1]
    w = width - 1
    h = height - 1
    dlx = x_max - x_min
    dly = y_max - y_min
    for point in data:
        x0 = int(w / dlx * (point[0] - x_min))
        y0 = int(h / dly * (point[1] - y_min))   # generated upside down
        y0 = h - (int(h / dly * (point[1] - y_min))) # generated normally
        img[y0,x0] = 255
    return img


def get_boundary(data):
    x = data[:,0]
    y = data[:,1]
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    return((x_min,y_min),(x_max,y_max))


# IFS
def generate_ifs(max_iter, params):

    probs = [i['p'] for i in params]
    x = 0
    y = 0
    points = np.zeros((max_iter,2))
    for i in range(max_iter):
        k = np.random.choice(len(probs),1,p=probs)[0]
        xn = params[k]['a'] * x + params[k]['b'] * y + params[k]['e']
        yn = params[k]['c'] * x + params[k]['d'] * y + params[k]['f']
        points[i] = (xn,yn)
        x = xn
        y = yn
    return points


# IFS  two-dimensional point histogram
def point_to_img(point,boundary,width,height):
    x_min, y_min = boundary[0]
    x_max, y_max = boundary[1]
    w = width - 1
    h = height - 1
    dlx = x_max - x_min
    dly = y_max - y_min
    x0 = int(w / dlx * (point[0] - x_min))
    y0 = h - int(h / dly * (point[1] - y_min))
    return  x0,y0


def data_to_img2(data,width,height):
    img = np.zeros((height,width),dtype = np.uint8)
    boudary = get_boundary(data)
    for point in data:
        x0,y0 = point_to_img(point,boudary,width,height)
        img[y0,x0] = 255
    return img


def hist_data(data, width,height):
    hist = np.zeros((width,height))
    boundary = get_boundary(data)
    for point in data:
        x0,y0 = point_to_img(point,boundary,width,height)
        hist[y0,x0] += 1
    return hist


def hist_to_img(hist):
    rows,cols = hist.shape
    img = np.zeros((rows,cols),dtype=np.uint8)
    max_value = hist.max()   # max = 255 | min = 0
    max_value = np.log(max_value)
    scale = (255/max_value)
    for row in range(rows):
        for col in range(cols):
            val = hist[row,col]
            if val != 0:
                val = scale * np.log(val)
            img[row,col] = int(val)
    return img

# sierpinski
# data = generate_sierpinski(1000)
# plt.plot(data[:,0], data[:,1], 'r*')
# plt.show()

# sierpinski jpg
# data = generate_sierpinski(1000)
# img = data_to_img(data,800,800)
# misc.imsave('s.jpg',img)

#IFS matplotlib
fern = [{'a':0,'b':0,'c':0,'d':0.16,'e':0,'f':0,'p':0.1},
        {'a': 0.2,'b': -0.26,'c': 0.23,'d':0.22,'e':0,'f':1.6,'p':0.08},
        {'a':-0.15,'b':0.28,'c':0.26,'d':0.24,'e':0,'f':0.44,'p':0.08},
        {'a':0.75,'b':0.04,'c':-0.04,'d':0.85,'e':0,'f':1.6,'p':0.74}]

fern2 = [{'a':0,'b':0,'c':0,'d':0.16,'e':0,'f':0,'p':0.01},
        {'a': 0.2,'b': -0.26,'c': 0.23,'d':0.22,'e':0,'f':1.6,'p':0.07},
        {'a':-0.15,'b':0.28,'c':0.26,'d':0.24,'e':0,'f':0.44,'p':0.07},
        {'a':0.85,'b':0.04,'c':-0.04,'d':0.85,'e':0,'f':1.6,'p':0.85}]

# points = generate_ifs(1000,fern)
# plt.plot(points[:,0],points[:,1],'r*')
# plt.show()


# IFS to jpg
points = generate_ifs(1000000,fern2)
img = data_to_img(points,800,800)
misc.imsave('fern.jpg',img)


# IFS two-dimensional histogram of jpg points
# data = generate_sierpinski(1000)
# img = data_to_img2(data,800,800)
# misc.imsave('histogram.jpg', img)

# IFS  two-dimensional point histogram
# data = generate_ifs(10000,fern2)
# hist = hist_data(data,800,800)
# plt.imshow(hist,cmap='terrain')
# plt.show()


# IFS histogram mapping to shades of gray
# data = generate_ifs(300000,fern2)
# hist = hist_data(data,800,800)
# img = hist_to_img(hist)
# misc.imsave('odcienie_szarosci.jpg', img)