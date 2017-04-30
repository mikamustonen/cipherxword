#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solves a cipher crossword puzzle from an image.
"""

from cipherxword import CipherCrossword
import argparse
import cv2


# Handle the command-line arguments
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("inputfile", nargs=1, type=str,
    help="the puzzle image file name")
parser.add_argument("outputfile", nargs=1, type=str,
    help="output file name for the solved puzzle")
parser.add_argument("--puzzle-border", type=str, default=None,
    help="output file for an image with the detected puzzle border drawn")
options = parser.parse_args()

# Read and solve the puzzle
cw = CipherCrossword(options.inputfile[0])
puzzle_with_border = cw.detect_puzzle(visualize=True)
if options.puzzle_border:
    cv2.imwrite(options.puzzle_border, puzzle_with_border)

puzzle = cw.read_puzzle(verbose=True)
solution, solved = cw.solve("data/words_finnish.txt", options.outputfile[0])

print("\nThe solution key:")
print(", ".join(["{}: {}".format(k, v.upper()) for k, v in solution.items()]))
print("\nThe solved puzzle:")
print(solved)
