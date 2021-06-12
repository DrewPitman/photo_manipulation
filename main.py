import numpy as np
from PIL import Image as im
import photo_manipulation as ph

if __name__ == "__main__":
    photo_name = "ED_01.jpg"
    pic = ph.color_image("photos/" + photo_name)
    pic = pic.get_grayscale(lambda r, g, b: b)
    # pic.r = ph.grayscale(np.zeros([pic.height,pic.width]))
    # pic = pic.get_grayscale(lambda r, g, b: (r+2*g+2*b)/5)
    # height = pic.height
    # width = pic.width
    # pic.scale(0.125)
    # pic.vec_blur(ph.gaus_vec(3))
    # pic = pic.sketch()

    # l = 0.75
    # pic.tint(lambda r, g, b: (l*b+(1-l)*r)-g/10,
    #          lambda r, g, b: np.maximum(g,l*b+(1-l)*r-g/10),
    #          lambda r, g, b: (l*b+(1-l)*r)-g/10)
    # pic.bokeh_blur(2)
    pic.show()
    pic.array = np.multiply(pic.get_fill(800,800,50),pic.array).astype(np.uint8)
    pic.show()
    pic.save("output/" + "ED_06.png")
