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
parser.add_argument("--digits", type=str, default=None,
    help="output file for an image with the detected digits drawn")
parser.add_argument("--digit-min-width", type=float, default=0.05,
    help="minimum width for a digit in cells")
parser.add_argument("--digit-max-width", type=float, default=0.4,
    help="maximum width for a digit in cells")
parser.add_argument("--digit-min-height", type=float, default=0.1,
    help="minimum height for a digit in cells")
parser.add_argument("--digit-max-height", type=float, default=0.5,
    help="maximum height for a digit in cells")
parser.add_argument("--filled-cell-threshold", type=float, default=0.8,
    help="threshold for a cell to be considered filled")
options = parser.parse_args()

# Read and solve the puzzle
cw = CipherCrossword(options.inputfile[0])
puzzle_with_border = cw.detect_puzzle(visualize=True)
if options.puzzle_border:
    cv2.imwrite(options.puzzle_border, puzzle_with_border)

print("Reading the puzzle from the image...")
puzzle = cw.read_puzzle(verbose=True, digit_min_width=options.digit_min_width,
                        digit_max_width=options.digit_max_width,
                        digit_min_height=options.digit_min_height,
                        digit_max_height=options.digit_max_height,
                        filled_cell_threshold=options.filled_cell_threshold)
if options.digits:
    digits_overlaid = cw.overlay(puzzle)
    cv2.imwrite(options.digits, digits_overlaid)

print("Solving the puzzle using simulated annealing...")
solution, solved = cw.solve("data/words_finnish.txt", options.outputfile[0])

print("\nThe solution key:")
print(", ".join(["{}: {}".format(k, v.upper()) for k, v in solution.items()]))
print("\nThe solved puzzle:")
print(solved)
