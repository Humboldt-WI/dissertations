import os
from gensim.models import Word2Vec
import random
from tqdm import tqdm

# Set seed to make results reproducible
random.seed(42)

BASE_PATH = "data/train/"
OUT_PATH = "data/embeddings"

datasets = ["amazon_laptop", "yelp_restaurant"]

# Create samples for 5, 10, ..., 100 of the dataset
sample_rates = range(5, 105, 5)

# Iterate 10 times
iterations = range(0, 10)

# Set env.PYTHONHASHSEED to 42. E.g. export PYTHONHASHSEED=42
# Remove out if not required
assert os.environ['PYTHONHASHSEED'] == '42', "Hash seed should be set to 42 to reproduce results"


def train_word2vec(corpus: list[str]):
    """Train a word to Word2Vec model"""
    return Word2Vec(corpus,
                    sg=0,  # default: 0, CBOW
                    window=10,  # default: 5
                    size=200,  # default: 100
                    min_count=2,  # default: 5
                    workers=1)  # default 3, set to 1 for reproducible results


def get_corpus_from_file(infile_path: str):
    """Read corpus from a file."""
    with open(infile_path, "r") as infile:
        return [x.lower().strip().split() for x in infile]


# Total iterations, merely used for progress bar
n_total = len(datasets) * len(sample_rates) * len(iterations)

with tqdm(total=n_total) as pbar:
    for dataset in datasets:
        infile_path = "{0}/{1}.txt".format(BASE_PATH, dataset)
        corpus = get_corpus_from_file(infile_path)

        for i in iterations:
            # For each iteration shuffle to corpus.
            # The same order is used for each sample within the iteration.
            random.shuffle(corpus)

            for sample_rate in sample_rates:
                outfile_path = "{0}/{1}-{2}-{3}.w2v".format(OUT_PATH, dataset, sample_rate, i)

                # Cut off list (corpus) at sample size
                corpus_sample = corpus[:round(len(corpus) * sample_rate / 100)]

                # Train and save the Word2Vec model
                model = train_word2vec(corpus_sample)
                model.wv.save_word2vec_format(outfile_path)

                pbar.update(1)
