import numpy as np
from PIL import Image as im
import PIL

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
print('Hello, World!')

picFile = im.open("photos/forest_clearing_01.jpg")
print(type("photos/forest_clearing_01.jpg") is str)
print(type("photos/forest_clearing_01.jpg"))
print(type(picFile))
print(type(picFile) is PIL.JpegImagePlugin.JpegImageFile)
pic = np.asarray(picFile)
print(pic.ndim)
pic = np.array(pic)

# pic = scale_down(pic)
# pic = scale_down(pic)

# [r, g, b] = separate(pic)
# r = two_pass_blur(r,
#                   [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 21, 22, 24, 25, 27, 29,
#                    30, 32, 33, 34, 35, 36, 37, 38, 39, 39, 39, 40, 39, 39, 39, 38, 37, 36, 35, 34, 33, 32, 30, 29, 27,
#                    25, 24, 22, 21, 19, 17, 16, 15, 13, 12, 11, 9, 8, 7, 7, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1])
# pic = integrate(r, g, b)

pic = rgb_to_gray(pic)


def semi_circle(x, y):
    if (x - 1) ** 2 + y ** 2 < 1:
        return 1
    else:
        return 0


semi_circle_mask = mask(pic, make_mask([len(pic), len(pic[0])], [2, 2], semi_circle))

A = np.array([[0, 0.76, 1, 0.76, 0],
              [0.76, 0.41, 0, 0.41, 0.76],
              [1, 0, 0, 0, 1],
              [0.76, 0.41, 0, 0.41, 0.76],
              [0, 0.76, 1, 0.76, 0]]) / 11.72

B = np.array([[0, 0.76, 1, 0.76, 0],
              [0.76, 1, 1, 0.41, 0.76],
              [1, 1, 1, 1, 1],
              [0.76, 1, 1, 1, 0.76],
              [0, 0.76, 1, 0.76, 0]]) / 18.5

C = np.array([[1, 5, 8, 5, 1],
              [5, 25, 40, 25, 5],
              [8, 40, 64, 40, 8],
              [5, 25, 40, 25, 5],
              [1, 5, 8, 5, 1]]) / 400

D = np.array([
    [0, 0, 0.5, 1, 0.5, 0, 0],
    [0, 0.75, 1, 1, 1, 0.75, 0],
    [0.5, 1, 1, 1, 1, 1, 0.5],
    [1, 1, 1, 1, 1, 1, 1],
    [0.5, 1, 1, 1, 1, 1, 0.5],
    [0, 0.75, 1, 1, 1, 0.75, 0],
    [0, 0, 0.5, 1, 0.5, 0, 0],
]) / 32

semi_circle_mask[1] = one_pass_blur(semi_circle_mask[1], D, [2, 2])
pic = np.add(semi_circle_mask[0], semi_circle_mask[1])
# pic = two_pass_blur(pic, [1, 5, 8, 5, 1], [1, 5, 8, 5, 1])
# pic = outline_edges(pic)
# pic = gray_to_rgb(pic)
# pic = 255 - pic
pic = pic.astype(np.uint8)
image = im.fromarray(pic)
image.save('forest_blur.png')
image.show()
