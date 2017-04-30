# -*- coding: utf-8 -*-

import random
from collections import Counter
from numpy import exp


class CryptogramSolver(object):
    """A cryptogram solver class, initialized with a (non-exhaustive) list of
    words in the target language. Uses simulated annealing.
    """
    
    def __init__(self, wordlist_file):
        
        # Store the known words to a set for O(1) lookup
        with open(wordlist_file, encoding='utf-8') as f:
            self.known_words = set(word.strip() for word in f.readlines())
        
        # Analyze the character n-grams for n=1,2,3
        self.characters = Counter(generate_ngrams(self.known_words, 1))
        self.bigrams = Counter(generate_ngrams(self.known_words, 2))
        self.trigrams = Counter(generate_ngrams(self.known_words, 3))
        
        # Constants weighting the number of found common bigrams and trigrams
        # in the scoring function
        self.bigram_const = 1.0/self.bigrams.most_common(1)[0][1]
        self.trigram_const = 1.0/self.trigrams.most_common(1)[0][1]
    
    
    def solve(self, cryptogram):
        """Solve a given cryptogram using simulated annealing.
        
        Args:
            cryptogram:  a list of encoded words
        
        Returns:
            a dictionary mapping the encoding of the cryptogram to the alphabet
        """
        
        # The initial guess is based on character frequencies in the language
        clues = Counter([x for word in cryptogram for x in word])
        numbers_by_frequency = [x for x, _ in clues.most_common()]
        letters_by_frequency = [x for x, _ in self.characters.most_common()]

        # Pad the numbers with ones that don't exist in puzzle, so that we
        # don't end up excluding the most uncommon letters from the potential
        # solutions
        needed_padding = len(letters_by_frequency) - len(numbers_by_frequency)
        numbers_by_frequency.extend([i + max(numbers_by_frequency) + 1
            for i in range(needed_padding)])
        
        # OCR errors might cause there to be extra numbers in the cryptogram;
        # Pad the key with stars in such a case
        if needed_padding < 0:
            letters_by_frequency.extend(["*" for i in range(-needed_padding)])
        key = {x: y for x, y in zip(numbers_by_frequency, letters_by_frequency)}
        
        # Initialize the simulated annealing
        trial = apply_key(cryptogram, key)
        initial_temperature = 100.0
        reheat_trigger = 2000
        no_progress_steps = 0
        temperature = initial_temperature
        max_iterations = 100000
        best_score = self.score_proposed_solution(trial)
        prev_score = best_score
        best_key = key.copy()
        initial_key = key.copy()
        
        # Annealing main loop
        for iteration in range(max_iterations):
            
            # A step in the random walk: Try swapping two entries in the key
            x, y = random.sample(key.keys(), 2)
            key[x], key[y] = key[y], key[x]
            score = self.score_proposed_solution(apply_key(cryptogram, key))
            
            # Check if this is better than any other solution encountered so far
            if score > best_score:
                best_score = score
                best_key = key.copy()
                print("{:06}  New best solution, score {}".format(iteration, score))
            
            # Metropolis-Hastings criterion for accepting the step
            if score <= prev_score:
                prob = exp(-(prev_score - score)/temperature)
            if score > prev_score or random.random() < prob:
                prev_score = score
                temperature *= 0.9
            else:
                # swap back
                key[x], key[y] = key[y], key[x]
                no_progress_steps += 1
    
            # When there has been no progress in a set number of steps, restart the
            # system to escape a possible local minimum
            if no_progress_steps == reheat_trigger:
                print("{:06}  Probably a local minimum, restarting".format(iteration))
                no_progress_steps = 0
                temperature = initial_temperature
                key = initial_key
                prev_score = self.score_proposed_solution(apply_key(cryptogram, key))
        
        # Filter out the padding we might have used
        best_key = {k: v for k, v in best_key.items() if k in clues}
        
        return best_key
        

    def score_proposed_solution(self, words):
        """The scoring function for a list of words, based on known words and
        character n-gram frequencies.
        """
        
        return (sum(len(x) for x in words if x in self.known_words)
            + self.bigram_const
                *sum(self.bigrams[x] for x in generate_ngrams(words, 2))
            + self.trigram_const
                *sum(self.trigrams[x] for x in generate_ngrams(words, 3)))


# Auxiliary functions

def generate_ngrams(words, n):
    """Produces all the character n-grams from the collection of words.
    """
    for word in words:
        if len(word) >= n:
            for i in range(len(word) - n + 1):
                yield word[i:i+n]


def apply_key(cryptogram, key):
    """Applies a key to the cryptogram.
    """
    return ["".join(key[x] for x in word) for word in cryptogram]
