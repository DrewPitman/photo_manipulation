import numpy as np
from PIL import Image as im


# for an input x, returns the nearest value to x on the interval [0, 255]
def limiter(x):
    return np.uint8(np.minimum(np.maximum(0, x), 255))


# the input array is a subarray of the output array, which has a border composed of  "row" extra rows
# and "col" extra columns before and after the subarray, filled with the value "value", which is 0 by default
def pad(array, row, col, value=0):
    padded_array = np.full([len(array) + 2 * row, len(array[0]) + 2 * col], value)
    padded_array[row:(len(array) + row), col:(len(array[0]) + col)] = array
    return padded_array


# given 3 grayscale objects representing the color fields of a single photo -- "r" representing red,
# "g" representing green, and "b" representing blue -- returns a color_image object with those color fields.
def grayscale_to_color(r, g, b):
    return color_image(
        np.array([[[r.array[i, j], g.array[i, j], b.array[i, j]] for j in range(r.width)] for i in range(r.height)]))


# 2-dimensional convolution of an input array with a kernel
# this operation centers the kernel at each entry of the input array
def convolve(array, kernel):
    pad_row = int(len(kernel) / 2) + 1
    pad_col = int(len(kernel[0]) / 2) + 1
    input_array = pad(array.astype(int), pad_row, pad_col)
    output_array = []
    # time tracking
    percent2 = -1
    for i in range(len(array)):
        percent1 = int(100 * i / len(array))
        if percent1 > percent2:
            print(str(percent1) + "%")
        percent2 = percent1
        #
        output_row = []
        for j in range(len(array[0])):
            subarray = input_array[i:i + len(kernel), j:j + len(kernel[0])]
            output_row.append(np.tensordot(kernel, subarray))
        output_array.append(output_row)
    return np.array(output_array)


# creates a kernel with a disk of ones in the center and zeros outside of that disk
# with anti-aliasing at the disk's edges
def bokeh_matrix(radius):
    matrix = np.zeros([2 * radius + 1, 2 * radius + 1])
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                matrix[i, j] = 1
            elif (i - radius) ** 2 + (j - radius) ** 2 < (radius + 1) ** 2:
                matrix[i, j] = np.sqrt((i - radius) ** 2 + (j - radius) ** 2) - radius
    normalization = sum(sum(matrix))
    return matrix / normalization


# returns a vector whose values follow a gaussian distribution
# the the first and last values lie 3 standard deviations away from the center value
def gaus_vec(radius):
    # radius is 3 times the standard deviation
    stdev = radius / 3
    gaussian = lambda x: np.exp(-x ** 2 / (2 * stdev))
    vector = np.array([[gaussian(i - radius) for i in range(2 * radius + 1)]])
    return vector / sum(sum(vector))

# a class representing grayscale images
class grayscale:
    def __init__(self, array):
        self.array = np.array(array).astype(np.uint8)
        self.height = len(array)
        self.width = len(array[0])

    # converts the grayscale image to a color_image object
    def get_color_image(self):
        return color_image(np.array([[[x, x, x] for x in A_row] for A_row in self.array]))

    # displays the image
    def show(self):
        self.get_color_image().show()

    # saves the image under the filename given by location
    def save(self, location):
        self.get_color_image().save(location)

    # changes the resolution of the image to one with height and width given as input
    def resolution(self, height, width):
        pic = []
        for i in range(height):
            pic_row = []
            row = int(i * self.height / height)
            for j in range(width):
                col = int(j * self.width / width)
                pic_row.append(self.array[row, col])
            pic.append(pic_row)
        self.array = np.array(pic).astype(np.uint8)
        self.height = len(pic)
        self.width = len(pic[0])

    # scales the resolution of the image by a multiplier given as input
    def scale(self, multiplier):
        self.resolution(int(multiplier * self.height), int(multiplier * self.width))

    # efficient blur operation returning a convolution of the image
    # using the outer product of an input vector with itself as the kernel
    def vec_blur(self, vector):
        # vectors must be 2d, e.g. np.array([[1,1,1]]) with two sets of brackets
        array = convolve(self.array, vector)
        array = convolve(self.array, vector.transpose())
        self.array = array.astype(np.uint8)

    # slower blur operation returning a convolution of the image with an input kernel
    def mat_blur(self, kernel):
        self.array = convolve(self.array, kernel).astype(np.uint8)

    # basic sobel edge detection returning a pair of arrays representing
    # convolutions of the image with the sobel array and its transpose
    def edge_detection(self):
        sobel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        vert_edges = convolve(self.array, sobel)
        horiz_edges = convolve(self.array, sobel.transpose())
        return [vert_edges, horiz_edges]

    # returns a color_image object which uses color to represent the output of edge_detection
    def rgb_edges(self):
        [vert_edges, horiz_edges] = self.edge_detection()
        green = np.full((self.height, self.width), 128)
        red = grayscale(limiter(green + vert_edges))
        blue = grayscale(limiter(green + horiz_edges))
        green = grayscale(green)
        return grayscale_to_color(red, blue, green)

    # detects the edges of the image and returns a grayscale object consisting of
    # a white background with black lines at the detected edges
    def sketch(self):
        [vert_edges, horiz_edges] = self.edge_detection()
        edges = np.sqrt(vert_edges ** 2 + horiz_edges ** 2)
        return grayscale(255 - limiter(edges))

    # slowly performs a bokeh blur via array convolution with a bokeh matrix whose disks have the given radius
    def bokeh_blur(self, radius):
        self.array.mat_blur(bokeh_matrix(radius))

    # converts the image into a mask where pixels carrying value "threshold" (default of 128) or higher
    # are converted to 1 and pixels of value lower than "threshold" are converted to 0.
    # users may supply a function to manipulate the pixel values before creating the mask
    def make_mask(self, threshold=128, func=lambda x: x):
        # func should return a value of threshold or higher if the mask is to contain a 1, and lower if 0
        return [[(func(self.array[i, j]) >= threshold) for j in range(self.width)] for i in range(self.height)]

    # given the location of a pixel, performs edge detection and returns an array of 1s and 0s
    # with 1s in an area surrounded by edges containing the input location
    # the input threshold indicates the required value of intensity of the edges
    def get_fill(self, x, y, threshold=128):
        fill_array = np.zeros([self.height + 2, self.width + 2])
        fill_array[x, y] = 1
        # First, we do edge detection
        [vert_edges, horiz_edges] = self.edge_detection()
        edges = np.sqrt(vert_edges ** 2 + horiz_edges ** 2)
        edges = pad(edges, 1, 1, 1020)

        pixel_list = [[x, y]]

        while pixel_list:
            pixel = pixel_list.pop()
            fill_array[pixel[0], pixel[1]] = 1
            for i in range(pixel[0] - 1, pixel[0] + 2):
                for j in range(pixel[1] - 1, pixel[1] + 2):
                    if edges[i, j] < threshold and fill_array[i, j] == 0 and [i, j] not in pixel_list:
                        pixel_list.append([i, j])

        return fill_array[1:-1, 1:-1]


# a class for rgb-valued images
class color_image:
    def __init__(self, photo):
        if type(photo) is np.ndarray:
            self.rgb = photo
        elif type(photo) is str:
            self.rgb = np.asarray(im.open(photo))
        else:
            self.rgb = np.asarray(photo)
        self.r = grayscale(self.rgb[:, :, 0])
        self.g = grayscale(self.rgb[:, :, 1])
        self.b = grayscale(self.rgb[:, :, 2])
        self.height = self.r.height
        self.width = self.r.width

    # displays the image
    def show(self):
        im.fromarray(self.rgb).show()

    # saves the image under the filename given by location
    def save(self, location):
        im.fromarray(self.rgb).save(location)

    # converts the image to a grayscale image by applying the function func to each pixel
    def get_grayscale(self, func=lambda r,g,b:(r+2*g+b)/4):
        # may need to rewrite in order to accommodate max and min functions
        gray_array = func(self.r.array.astype(int), self.g.array.astype(int),
                          self.b.array.astype(int))
        return grayscale(limiter(gray_array))

    # deletes self.rgb and reforms it from self.r, self.g, and self.b
    def refresh_rgb(self):
        self.rgb = np.array(
            [[[self.r.array[i, j], self.g.array[i, j], self.b.array[i, j]] for j in range(self.r.width)] for i in
             range(self.r.height)]).astype(np.uint8)
        self.height = self.r.height
        self.width = self.r.width

    # changes the resolution of the image to one with height and width given as input
    def resolution(self, height, width):
        self.r.resolution(height, width)
        self.g.resolution(height, width)
        self.b.resolution(height, width)
        self.refresh_rgb()

    # scales the resolution of the image by a multiplier given as input
    def scale(self, multiplier):
        self.r.scale(multiplier)
        self.g.scale(multiplier)
        self.b.scale(multiplier)
        self.refresh_rgb()

    # Takes 3 functions as input. Each function takes 3 numerical inputs and gives one numerical output.
    # Each of these functions is used to produce a distinct grayscale image from the original rgb color image.
    # these 3 grayscale images then become the r, g, and b fields of the object.
    def tint(self, rfunc, gfunc, bfunc):
        red = self.get_grayscale(rfunc)
        green = self.get_grayscale(gfunc)
        blue = self.get_grayscale(bfunc)
        self.r = red
        self.g = green
        self.b = blue
        self.refresh_rgb()

    # slowly performs a bokeh blur via array convolution with a bokeh matrix whose disks have the given radius
    def bokeh_blur(self, radius):
        self.r.mat_blur(bokeh_matrix(radius))
        self.g.mat_blur(bokeh_matrix(radius))
        self.b.mat_blur(bokeh_matrix(radius))
        self.refresh_rgb()
