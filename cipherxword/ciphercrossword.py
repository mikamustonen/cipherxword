# -*- coding: utf-8 -*-

import cv2
import numpy as np


class CipherCrossword(object):
    """This class takes an image file containing a cipher crossword puzzle
    and encapsulates operations for extracting the puzzle and solving it.
    """
    
    def __init__(self, filename):
        
        # If OpenCV fails to read the image (say, because the path was wrong),
        # imread returns None.
        self.image_original = cv2.imread(filename)
        if self.image_original is None:
            raise RuntimeError("Unable to read the image file.")
        self.image_grayscale = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2GRAY)
    
    
    def detect_puzzle(self, visualize=False):
        """Detects the puzzle from the image, assuming it is bounded by
        the largest contour by area.
        
        Args:
            visualize: If True, returns the image with the identified puzzle
                marked. (optional, default: False)
        """
        # Threshold the image and find the largest contour by area
        self.image_thresholded = cv2.adaptiveThreshold(self.image_grayscale,
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 15)
        # findContours modifies the source image, so make a copy for it
        image = np.copy(self.image_thresholded)
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)
        self.puzzle_border = max(contours, key=cv2.contourArea)
        
        if visualize:
            image = np.copy(self.image_original)
            cv2.drawContours(image, [self.puzzle_border], -1, (0,0,255), 3)
            return image
    
    
    def read_puzzle(self, verbose=False):
        """Reads the puzzle into an array.
        
        Returns:
            a NumPy array of integers with the number recognized from
            the square for empty cells, and -1 for filled cells.
        """
        # First determine how many cells we have in each direction by looking
        # at the brightness profile of the thresholded image within the puzzle
        # area
        x, y, w, h = cv2.boundingRect(self.puzzle_border)
        self.puzzle_area = self.image_thresholded[y : y + h, x : x + w]
        self.width_in_cells = count_cells(np.average(self.puzzle_area, axis=0))
        self.height_in_cells = count_cells(np.average(self.puzzle_area, axis=1))
        
        if verbose:
            print("Detected puzzle dimension: ({}, {})".format(
                self.width_in_cells, self.height_in_cells))
        
        # TODO: Detect the filled cells
        
        # TODO: OCR the digits on the empty cells

    
    def overlay(self, values):
        """Overlays an array of strings or integers on top of the original
        image of the puzzle.
        
        Args:
            values: a two-dimensional array of strings or integers to be
                overlayed on the image.
        
        Returns:
            the original image with the overlayed data
        """
        raise NotImplementedError()


# Auxiliary functions
def count_cells(profile, min_cells=2, max_cells=50):
    """Counts the number of equal-sized cells from a brightness profile,
    assuming that the whole profile is covered with cells, by comparing
    the profile to an idealized profile with different numbers of cells.
    
    Args:
        profile: a NumPy array containing the brightness profile; cell borders
            are expected to be brighter than cells on average (negative image)
        min_cells: (optional) the minimum number of cells to consider
        max_cells: (optional) the maximum number of cells to consider
    
    Returns:
        The number of cells detected.
    """
    # First, threshold the profile
    profile[profile < 128] = 0
    profile[profile >= 128] = 255
    
    # There's no need to normalize the profile, as long as we normalize
    # the idealized profile
    goodness = np.array([np.dot(profile, idealized_profile(i, len(profile)))
        for i in range(min_cells, max_cells + 1)])
    best_match = np.argmax(goodness) + min_cells
    
    return best_match


def idealized_profile(n, length):
    """An idealized brightness profile for n cells.
    
    Args:
        n:      the number of cells
        length: length of the requested profile array
    
    Returns:
        A NumPy array with Gaussian peaks placed on equidistant intervals
    """
    
    profile = np.zeros(length)
    cell_width = length/n
    gaussian_width = 0.1*cell_width
    x = np.array(range(length))
    
    # Leave out the end points to emphasize match on the cell borders
    for i in range(1, n):
        x0 = i*length/n
        profile += np.exp(-(x - x0)**2/(2*gaussian_width**2))
    
    # Normalize
    profile /= np.sum(profile**2)
    
    return profile
