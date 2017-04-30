# -*- coding: utf-8 -*-

import cv2
import numpy as np
from cipherxword.digitclassifier import DigitClassifier


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
        self.digit_classifier = DigitClassifier()
    
    
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
    
    
    def read_puzzle(self, verbose=False, digit_min_width=0.05,
        digit_max_width=0.4, digit_min_height=0.1, digit_max_height=0.5):
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
        self.puzzle_x, self.puzzle_y = x, y
        self.puzzle_width, self.puzzle_height = w, h
        
        if verbose:
            print("Detected puzzle dimension: ({}, {})".format(
                self.width_in_cells, self.height_in_cells))
        
        # By default, every square is empty or outside the puzzle
        self.puzzle = -np.ones((self.width_in_cells, self.height_in_cells),
            dtype=np.int16)
        
        # An educated guess for filled_threshold: 80% of the maximum brightness
        filled_threshold = 0.8*np.max([np.average(im)
            for _, _, im in self._cell_images()])
        
        # Read the numbers in the empty squares
        # First, translate the reasonable digit size to pixels
        average_square_width = self.puzzle_width/self.width_in_cells
        average_square_height = self.puzzle_height/self.height_in_cells
        wmin = digit_min_width*average_square_width
        wmax = digit_max_width*average_square_width
        hmin = digit_min_height*average_square_height
        hmax = digit_max_height*average_square_height
        
        for ix, iy, cell_image in self._cell_images():
            # The cell images are taken from the thresholded image, so they
            # are negative: filled squares hence appear bright
            if np.average(cell_image) < filled_threshold:
                image = 255 - cell_image # flip the negative
                _, contours, _ = cv2.findContours(cell_image, cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE)
                # Filter contours that are in a reasonable rage of height and
                # width
                digits = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if wmin <= w <= wmax and hmin <= h <= hmax:
                        digits.append(c)
                
                # At this point, our set of contours might include ones that
                # are holes in the numbers, instead of holes; figure out which
                # ones are likely actual digits by looking at pairwise overlaps
                # of their bounding boxes
                bboxes = [cv2.boundingRect(c) for c in digits]
                ok = [True for c in contours]
                for x1, y1, w1, h1 in bboxes:
                    for i, (x2, y2, w2, h2) in enumerate(bboxes):
                        # If the horizontal overlap of the two bounding boxes
                        # is more than two pixels, and bounding box 1 is larger,
                        # mark bounding box 2 for removal
                        if (max(min(x1 + w1, x2 + w2) - max(x1, x2), 0) > 2 and
                            h2*w2 < h1*w1): ok[i] = False
                
                bboxes = [b for i, b in enumerate(bboxes) if ok[i]]
                bboxes.sort(key=lambda x: -x[0]) # order from right to left
                
                # Classify each digit and combine them to a number
                digit_values = self.digit_classifier.predict([
                    image[y:y+h,x:x+w] for x, y, w, h in bboxes])
                self.puzzle[ix,iy] = sum(d*10**pos
                    for pos, d in enumerate(digit_values))
        
        return self.puzzle 

    
    def _cell_images(self):
        """A generator that yields the cells in the puzzle.
        
        Works only after the puzzle dimensions in cells are determined. Meant
        to be called from read_puzzle.
        """
        average_width = float(self.puzzle_width)/self.width_in_cells
        average_height = float(self.puzzle_height)/self.height_in_cells
        
        for ix in range(self.width_in_cells):
            for iy in range(self.height_in_cells):
                # The puzzle is not necessarily rectangular, so check that
                # we are inside the puzzle borders
                x1, y1 = int(ix*average_width), int(iy*average_height)
                x2, y2 = int((ix+1)*average_width), int((iy+1)*average_height)
                midpoint = (self.puzzle_x + int((ix + 0.5)*average_width),
                    self.puzzle_y + int((iy + 0.5)*average_height))
                if cv2.pointPolygonTest(self.puzzle_border, midpoint, False) > 0:
                    square = self.puzzle_area[y1:y2, x1:x2]
                    yield ix, iy, square
    
    
    def overlay(self, values, color=(0,0,255), blank=-1):
        """Overlays an array of strings or integers on top of the original
        image of the puzzle.
        
        Args:
            values: a two-dimensional array of strings or integers to be
                overlayed on the image.
        
        Returns:
            the original image with the overlayed data
        """
        image = np.copy(self.image_original)
        for ix in range(self.width_in_cells):
            for iy in range(self.height_in_cells):
                if values[ix][iy] != blank:
                    loc = (int(self.puzzle_x + (ix + 0.25)*self.puzzle_width/self.width_in_cells),
                        int(self.puzzle_y + (iy + 0.8)*self.puzzle_height/self.height_in_cells))
                    cv2.putText(image, str(values[ix][iy]), loc,
                        cv2.FONT_HERSHEY_COMPLEX, 1, color)
        
        return image


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
