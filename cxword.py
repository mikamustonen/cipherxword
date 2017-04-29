#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solves a cipher crossword puzzle from an image.
"""

from cipherxword import CipherCrossword
import argparse


# Handle the command-line arguments
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("inputfile", nargs=1, type=str,
    help="the puzzle image file name")
options = parser.parse_args()

# Read and solve the puzzle
cw = CipherCrossword(options.inputfile[0])
