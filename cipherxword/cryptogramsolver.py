# -*- coding: utf-8 -*-


class CryptogramSolver(object):
    """A cryptogram solver class, initialized with a (non-exhaustive) list of
    words in the target language. Uses simulated annealing.
    """
    
    def __init__(self, wordlist_file):
        raise NotImplementedError()
    
    
    def solve(self, encrypted_words):
        raise NotImplementedError()
