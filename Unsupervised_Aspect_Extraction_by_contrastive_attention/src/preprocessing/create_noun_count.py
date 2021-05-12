# Collect all nouns from the training corpora.
# Nouns are sorted by their frequency and written to a textfile.

import collections
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

DATA_TRAIN_BASE_PATH = "data/train/"
NOUN_COUNT_BASE_PATH = "data/noun_count/"

datasets = ['amazon_laptop', 'yelp_restaurant']


def is_noun(pos_tag: str) -> bool:
    """Test if token is a noun"""
    return pos_tag == 'NOUN'


def get_nouns_from_string(line: str) -> list[str]:
    """Get nouns from a string using POS tagger provided by spaCy."""
    return [ent.text for ent in nlp(line) if is_noun(ent.pos_)]


def get_nouns_from_doc(doc: list[str]) -> list[str]:
    """Get all nouns from a document. """
    nouns = []

    for line in tqdm(doc):
        nouns_in_line = get_nouns_from_string(line)
        nouns.extend(nouns_in_line)
    return nouns


for dataset in datasets:
    dataset_path = DATA_TRAIN_BASE_PATH + dataset + ".txt"
    nouns_sorted_path = NOUN_COUNT_BASE_PATH + dataset + ".txt"

    with open(dataset_path, "r") as infile:
        lines = infile.read().splitlines()

    # List of all nouns in document
    nouns_in_doc = get_nouns_from_doc(lines)

    # Get frequency of distinct nouns
    noun_count = collections.Counter(nouns_in_doc)

    # Get descending, sorted list of nouns by frequency
    nouns_in_doc_sorted = sorted(noun_count, key=noun_count.get, reverse=True)

    with open(nouns_sorted_path, 'w') as f:
        for noun in nouns_in_doc_sorted:
            f.write("%s\n" % noun)
