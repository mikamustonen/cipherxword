# -*- coding: utf-8 -*-

from cipherxword import dataset
import numpy as np
import cv2
from sklearn import svm, metrics


class DigitClassifier(object):
    """A classifier for identifying printed digits.
    """
    
    def __init__(self, validation_set_size=30, verbose=False):
        # When creating the classifier, train it. Training a support vector
        # machine is so fast it makes no sense to store the fitted model on
        # the disk.
        digit_images, target = dataset.load_digits()
        features = features_pixels(digit_images)
        self.train(features, target, validation_set_size, verbose)

    
    def train(self, features, target, validation_set_size, verbose):
        """Trains the classifier. Automatically called when creating the object.
        
        Args:
            features:             the list of samples
            target:               the labels for the samples
            validation_set_size:  the number of samples to use for validation
            verbose:              if True, print the validation report
        """
        # Split the set into the training and validation set
        training_set = features[:-validation_set_size]
        training_target = target[:-validation_set_size]
        validation_set = features[-validation_set_size:]
        expected = target[-validation_set_size:]
        
        # Train a support vector machine
        self.classifier = svm.SVC(kernel='linear', gamma=0.01)
        self.classifier.fit(training_set, training_target)
        
        # Optionally print out the validation results
        if verbose:
            print("Digit classifier trained using {} samples.".format(
                len(training_set)))
            predicted = self.classifier.predict(validation_set)
            print("\nDigit classifier validation report:")
            print(metrics.classification_report(expected, predicted))
            print("\nConfusion matrix:")
            print(metrics.confusion_matrix(expected, predicted))
    
    
    def predict(self, images):
        """Identifies the digits in given grayscale images.
        
        Args:
            images: a list of images of digits
        """
        # Extract the features
        features = features_pixels(images)
        
        return self.classifier.predict(features)


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
    """Translates grayscale images to vectors of brightness values,
    similar to the sklearn MNIST dataset.
    
    Args:
        images: a list of grayscale images
    
    Returns:
        the list of 64-element vectors
    """
    features = [cv2.resize(preprocess_image(image), (12, 12)).flatten()
        for image in images]
    return features
