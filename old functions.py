sigmoid = lambda z: 255 / (1 + np.exp((127 - z) / 32))

def gray_val(pic, i, j):
    r = int(pic[i, j, 0])
    g = int(pic[i, j, 1])
    b = int(pic[i, j, 2])
    # return 255-int(10.63*(np.log2(256-r)+np.log2(256-g)+np.log2(256-b)))
    # return min(max(0,int(max(r-g-b,-r+g-b,-r-g+b,0)+0.5*r+0.3*g+0.2*b-25)),255)
    bright = max(r-g-b,-r+g-b,-r-g+b,0)
    bright_cross = max(r*g-r*b-g*b,-r*g+r*b-g*b,-r*g-r*b+g*b,0) >> 9
    white = (r*g*b) >> 17
    square = (r**2 + g**2 + b**2) >> 11
    cross_term = (r*g + r*b + g*b) >> 12
    linear = (r + 4*g + 0*b) >> 3
    # return int(sigmoid(1.5*(bright + bright_cross + white - square - cross_term + linear)))
    # return int(sigmoid(bright))
    # return int(sigmoid(bright_cross))
    # return int(sigmoid(white))
    # return int(sigmoid(square))
    # return int(sigmoid(cross_term))
    return int(sigmoid(linear))


def gray(pic, i, j):
    return [gray_val(pic, i, j), gray_val(pic, i, j), gray_val(pic, i, j)]


def negative(pic, i, j):
    return [255 - pic[i, j, 0], 255 - pic[i, j, 1], 255 - pic[i, j, 2]]


def blur(pic, i, j):
    h = len(pic)
    w = len(pic[0])
    red = 0
    green = 0
    blue = 0
    for iter in range(2):
        for jter in range(2):
            red += pic[(i + iter) % h, (j + jter) % w, 0]
            green += pic[(i + iter) % h, (j + jter) % w, 1]
            blue += pic[(i + iter) % h, (j + jter) % w, 2]
    return [int(red / 4), int(green / 4), int(blue / 4)]


def filter_dark(pic, i, j):
    h = len(pic)
    w = len(pic[0])
    if gray_val(pic, i, j) < 150:
        return [0, 0, 0]
    else:
        return pic[i, j]


def apply_effect(picture, f):
    pic = np.array(picture)
    a = pic
    iter = 0
    percent2 = -1
    for i in range(len(pic)):
        iter += 1
        # print(str(iter)+"/"+str(len(pic)))
        percent1 = int(100 * iter / len(pic))
        if percent1 > percent2:
            print(str(percent1) + "%   " + str(iter) + "/" + str(len(pic)))
        percent2 = percent1
        for j in range(len(pic[0])):
            a[i, j] = f(pic, i, j)
    return a