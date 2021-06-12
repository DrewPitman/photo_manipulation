# photo_manipulation
Edit photos with user-defined functions

## Motivation
photo_manipulation.py is a Python module containing classes and functions useful for photo editing.
Modern photo editing software typically does not allow the user to define numeric functions to be applied to the RGB values of an image's pixels.
For example, suppose you want to:
- create a grayscale image from a color image by multiplying the red values by the green values, adding the blue values, and then normalizing;
- replace the red values of an image with the green values and vice versa; or
- define the kernel of a convolution to be applied to an image
For these and more general operations, photo editing software seems to be useless.
photo_manipulation.py fills these gaps by providing classes for grayscale and color images along with methods and functions 
that can be used to numerically manipulate pixel intensity values.

## Requirements
You may need Python 3.9 or later.
You will need to install the PIL and Numpy libraries.

## Installation
Download photo_manipulation.py and add it to your path or working folder. 
