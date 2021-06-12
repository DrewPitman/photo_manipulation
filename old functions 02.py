# blur


def sobel(pic, i, j):
    x_mat = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_mat = x_mat.transpose()
    pic_slice = pic[i - 1:i + 2, j - 1:j + 2]
    return min(255, np.sqrt(np.tensordot(x_mat, pic_slice) ** 2 + np.tensordot(y_mat, pic_slice) ** 2))


def padding(pic, border):
    padded_pic = np.zeros([len(pic) + 2 * border, len(pic[0]) + 2 * border])
    padded_pic[border:(len(pic) + border), border:(len(pic[0]) + border)] = pic
    return padded_pic


def outline_edges(pic):
    pic = padding(pic, 1)
    outline = []
    percent2 = -1
    for i in range(1, len(pic) - 1):
        percent1 = int(100 * i / len(pic))
        if percent1 > percent2:
            print(str(percent1) + "%")
        percent2 = percent1
        outline_row = []
        for j in range(1, len(pic[0]) - 1):
            outline_row.append(sobel(pic, i, j))
        outline.append(outline_row)
    return np.array(outline).astype(np.uint8)


def separate(pic):
    return [pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]]


def integrate(r, g, b):
    return np.array([[[r[i, j], g[i, j], b[i, j]] for j in range(len(r[0]))] for i in range(len(r))])


def gray_to_rgb(gray, rgb=[1, 1, 1]):
    return integrate(rgb[0] * gray, rgb[1] * gray, rgb[2] * gray)


def rgb_to_gray(pic, rgb=[1, 1, 1]):
    rgb = np.array(rgb)
    rgb = rgb / sum(rgb)
    return rgb[0] * pic[:, :, 0] + rgb[1] * pic[:, :, 1] + rgb[2] * pic[:, :, 2]


def slice(pic, i, j, start, end, axis=0):
    if axis == 1:
        slice_elem = np.zeros(max(0, -j - start))
        slice_elem = np.append(slice_elem, pic[i, max(0, j + start):(min(len(pic[0]), j + end) + 1)])
        slice_elem = np.append(slice_elem, np.zeros(max(0, j + end - len(pic[0]) + 1)))
    elif axis == 0:
        slice_elem = np.zeros(max(0, -i - start))
        slice_elem = np.append(slice_elem, pic[max(0, i + start):(min(len(pic), i + end) + 1), j])
        slice_elem = np.append(slice_elem, np.zeros(max(0, i + end - len(pic) + 1)))
    return np.array(slice_elem)


def one_pass_blur(gray, A, a):
    # A is the kernel kernel, a tells you where to position A for a given pixel
    # current pixel
    A = np.array(A)
    l = max(len(A), len(A[0]))
    gray_pad = padding(gray, l)

    blur = []
    percent2 = -1
    for i in range(len(gray)):
        percent1 = int(100 * i / len(gray))
        if percent1 > percent2:
            print(str(percent1) + "%, first half")
        percent2 = percent1
        blur_row = []
        for j in range(len(gray[0])):
            B = gray_pad[i + l - a[0]:i + l - a[0] + len(A), j + l - a[1]:j + l - a[1] + len(A[1])]
            blur_row.append(np.tensordot(A, B))
        blur.append(blur_row)
    return np.array(blur)


def two_pass_blur(gray, a, b):
    a = np.array(a)
    b = np.array(b)
    a_len = len(a)
    a_ext = int(a_len / 2)
    b_len = len(b)
    b_ext = int(b_len / 2)
    b = b.transpose()

    pass1 = []

    percent2 = -1
    for i in range(len(gray)):
        percent1 = int(100 * i / len(gray))
        if percent1 > percent2:
            print(str(percent1) + "%, first half")
        percent2 = percent1
        pass_row = []
        for j in range(len(gray[0])):
            pass_vec = slice(gray, i, j, -a_ext, a_ext, 1)
            pass_row.append(a.dot(pass_vec.transpose()) / sum(a))
        pass1.append(pass_row)
    pass1 = np.array(pass1)

    pass2 = []
    for i in range(len(pass1)):
        percent1 = int(100 * i / len(gray))
        if percent1 > percent2:
            print(str(percent1) + "%, second half")
        percent2 = percent1
        pass_row = []
        for j in range(len(pass1[0])):
            pass_vec = slice(pass1, i, j, -b_ext, b_ext, 0)
            pass_row.append(b.dot(pass_vec.transpose()) / sum(b))
        pass2.append(pass_row)
    return np.array(pass2)


def scale_down(picture):
    a = []
    for i in range(len(picture)):
        if i % 2 == 0:
            b = []
            for j in range(len(picture[0])):
                if j % 2 == 0:
                    b.append(picture[i, j])
            a.append(b)
    return np.array(a)


def mask(gray, matrix):
    # A is a binary kernel that makes up the mask we want to return the masked image and the unmasked part
    return [np.multiply(gray, matrix), np.multiply(gray, 1 - matrix)]


def make_mask(resolution, dimensions, func):
    transform = lambda x, y: [y * dimensions[0] / resolution[1], (resolution[0] - x) * dimensions[1] / resolution[0]]
    mask_mat = []
    for i in range(resolution[0]):
        mask_row = []
        for j in range(resolution[1]):
            mask_row.append(func(transform(i, j)[0], transform(i, j)[1]))
        mask_mat.append(mask_row)
    return np.array(mask_mat)
