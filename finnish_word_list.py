#!/usr/bin/env python

"""This script extracts the list of Finnish words from the XML file provided by
the Institute for the Languages of Finland website. Place the XML file to the
folder data/ before running this.

The XML file can be downloaded from:
    http://kaino.kotus.fi/sanat/nykysuomi/kotus-sanalista-v1.tar.gz
"""

import xml.etree.ElementTree as etree
import re


input_filename = "data/kotus-sanalista_v1.xml"
output_filename = "data/words_finnish.txt"


tree = etree.parse(input_filename)
root = tree.getroot()

# Make sure all the words are lowercase and remove any special characters such
# as dashes -- as those are not valid in the crosswords. 
forbidden_characters = "[^a-zåäö]"
words = [re.sub(forbidden_characters, '', s.text.lower())
    for s in root.findall('.//st//s')]

# Save the wordlist
with open(output_filename, 'w', encoding='utf8') as f:
    for word in words:
        f.write("{}\n".format(word))
