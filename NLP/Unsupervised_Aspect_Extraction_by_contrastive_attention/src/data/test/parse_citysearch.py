# Simple Regex-based parser for the Citysearch dataset.
# The dataset is wrapped is some malformed XML, thus regular XML parsers (e.g. lxml) are
# not able to process it.
#
# Sentences have the format
# <Staff><Positive><5> It 's easy to get a table for a large group and you do n't get hustled out .</5></Positive></Staff>
#
# This script extracts all sentences with a single label.

import re
from data.helper_functions import strip_lower_tokenize


# Input and output files
citysearch_in_path = 'data/raw/test/citysearch_test.xml'
citysearch_out_path = 'data/test/citysearch.txt'
citysearch_out_labels_path = 'data/test/citysearch_labels.txt'

# Regex pattern to match
sentence_pattern = r'<[0-9]>(.+)<\/[0-9]>'

sentences_labels_list = []

labels = {'Price', 'Miscellaneous', 'Food', 'Anecdotes', 'Staff', 'Ambience'}

with open(citysearch_in_path, 'r') as infile:
    # Iterate over input file
    for line in infile:
        # Find labels in line using the predefined set
        line_labels = [s for s in labels if "<" + s + ">" in line]

        # Ignore line if not exactly one category is present.
        if len(line_labels) != 1:
            continue

        # Match the sentence inside de numbered XML tags.
        # Make use of the new named expressions introduced by Python 3.9
        # https://www.python.org/dev/peps/pep-0572/
        if (match := re.search(sentence_pattern, line)) is not None:
            label = line_labels[0]
            sentence = match.group(1)
            sentence = strip_lower_tokenize(sentence)
            t = (label, sentence)
            sentences_labels_list.append(t)


with open(citysearch_out_path, 'w') as sent_out, open(citysearch_out_labels_path, 'w') as lab_out:
    # Loop over the list of sentences and labels and write to separate files.
    for lab, sent in sentences_labels_list:
        lab_out.write("%s\n" % lab)
        sent_out.write("%s\n" % sent)
