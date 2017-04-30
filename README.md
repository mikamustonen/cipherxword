Finnish cipher crossword puzzle solver
======================================

This program takes an image of a Finnish cipher crossword and solves it using
simulated annealing.

The program is written in Python 3 and depends on OpenCV 3, scikit-learn, and
NumPy. It also requires a list of Finnish words, which can be obtained from
the [Institute for the Languages of Finland website](http://kaino.kotus.fi/sanat/nykysuomi/)
([a direct link to the file](http://kaino.kotus.fi/sanat/nykysuomi/kotus-sanalista-v1.tar.gz))
in the XML form. The word list can be extracted from the XML file using the
included script finnish_word_list.py, which expects the XML file to be located
in the subdirectory data/ and also places the extracted word list there.

The program is tested on macOS Sierra with the Anaconda distribution of
Python 3.5.

Usage
-----

A good test puzzle can be found [here](http://sanaris.fi/wordpress/images/tekijat/laatijat/antti_viitamaki_esimerkkisokkokrypto.jpg).
To solve that puzzle, download it to the program directory and try:

    ./cxword.py antti_viitamaki_esimerkkisokkokrypto.jpg solved.png

The second argument is the file name to which the solved puzzle should be
drawn.


Known limitations
-----------------

The input image must have a reasonable resolution and be upright, as no
perspective correction is applied. Google image search finds several puzzles
to try with the Finnish search terms "krypto ristikko". The puzzle is expected
to be the largest continuous object in the image.

Furthermore, the program is limited to the subtype of cipher crosswords with
no additional clues, i.e. no letters given and no photo clues.
