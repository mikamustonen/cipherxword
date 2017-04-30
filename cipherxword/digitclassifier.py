# -*- coding: utf-8 -*-

from cipherxword import dataset
import numpy as np
import cv2


class DigitClassifier(object):
    """A classifier for identifying printed digits.
    """
    
    def __init__(self, validation_set_size=30):
        # When creating the classifier, train it. Training a support vector
        # machine is so fast it makes no sense to store the fitted model on
        # the disk.
        digit_images, target = dataset.load_digits()
        features = features_pixels(digit_images)

    
    def train(self, features):
        raise NotImplementedError()
    
    
    def predict(self):
        raise NotImplementedError()


    def extract_features_pixels(self, images):
        """
        """


# Auxiliary functions
def preprocess_image(image):
    """Preprocesses a grayscale image by padding it to a square (original image
    centered) and adjusting the range of values to cover the whole interval
    0...255.
    
    Args:
        image:  a grayscale image
    
    Returns:
        the preprocessed image
    """
    
    # Compute the padding needed for each side to make image a square
    h, w = image.shape
    dim = max(h,w)
    if dim % 2 == 1: dim += 1
    padh, padw = (dim - h)//2, (dim - w)//2
    paddings = ((padh, dim - padh - h), (padw, dim - padw - w))
    
    # Pad with white and blur a little to reduce noise
    padded_image = np.pad(image, paddings, 'constant', constant_values=255)
    blurred_image = cv2.GaussianBlur(padded_image, (3,3), 0)
    
    # Adjust the range
    processed_image = blurred_image - np.min(blurred_image)
    processed_image = 255/np.max(processed_image)*processed_image
    
    return processed_image


def features_pixels(images):
    """Translates grayscale images to 64-value vectors of brightness values,
    similar to the sklearn MNIST dataset.
    
    Args:
        images: a list of grayscale images
    
    Returns:
        the list of 64-element vectors
    """
    features = [cv2.resize(preprocess_image(image), (8, 8)).flatten()
        for image in images]
    return features
