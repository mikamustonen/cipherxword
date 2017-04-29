# -*- coding: utf-8 -*-

import cv2


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
    
    
    def isolate_puzzle(self, visualize=False):
        """Identifies the puzzle from the image, assuming it is bounded by
        the largest contour by area.
        
        Args:
            visualize: If True, returns the image with the identified puzzle
                marked. (optional, default: False)
        """
        raise NotImplementedError()
    
    
    def read_puzzle(self):
        """Reads the puzzle into an array.
        
        Returns:
            a NumPy array of integers with the number recognized from
            the square for empty cells, and -1 for filled cells.
        """
        raise NotImplementedError()
    
    
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
