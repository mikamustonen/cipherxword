# -*- coding: utf-8 -*-

"""This module contains functions for reading the images of digits written
on a pure white background, in the order 0...9 on each row, separated by
whitespace.
"""

import cv2


digit_image_filenames = ["data/digitset{}.png".format(i) for i in range(1,5)]


def load_digits():
    """Reads the digit images from the default set of files.
    
    Returns:
        images: the digit images (grayscale, variable dimensions)
        target: an array of integers specifying what digit each image represents
    """
    
    images, target = [], []
    for image_file in digit_image_filenames:
        image = cv2.imread(image_file)
        if image is None:
            raise RuntimeError("Failed to read the image file '{}'".format(
                image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for slice in image_slices(image, 0):
            for i, character in enumerate(image_slices(slice, 1)):
                target.append(i)
                images.append(character)
    
    return images, target


def image_slices(image, axis=0):
    """A generator that yields the given grayscale image by split up on pure
    white along the chosen axis.
    """
    LOOKING_FOR_START, LOOKING_FOR_END = range(2)
    h = image.shape[axis]
    state = LOOKING_FOR_START
    for i in range(h):
        line = image[i,:] if axis == 0 else image[:,i]
        if state == LOOKING_FOR_START:
            if any(line != 255):
                istart = i
                state = LOOKING_FOR_END
        else:
            if all(line == 255):
                state = LOOKING_FOR_START
                if axis == 0:
                    yield image[istart:i,:]
                else:
                    yield image[:,istart:i]
