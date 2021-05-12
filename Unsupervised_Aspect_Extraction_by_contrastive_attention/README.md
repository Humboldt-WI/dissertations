# Unsupervised Aspect Extraction for Sentiment Analysis using Contrastive Attention

Code for the Bachelor's thesis "Unsupervised Aspect Extraction for Sentiment Analysis using Contrastive Attention".
Preprocessing and classification is performed in Python. Further analysis is done in R.
The code has been tested using Python 3.9.1 and R 4.0.3 on Linux. 

This code builds on the framework [CAt](https://github.com/clips/cat), developed by Stéphan Tulkens and Andreas van Cranenburgh.

# Howto

## Install Python dependencies

1. Install CAt. A packaged version is located at `src/resources/python/cat-0.1.tar.gz` 
   In a shell execute `pip install src/resources/python/cat-0.1.tar.gz`.
2. The requirements file can be used to install further Python dependencies.
   In a shell execute `pip install -r requirements.txt`.
3. Install a trained Spacy pipeline for the English language. We used [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm).
   `python -m spacy download en_core_web_sm`

## Install R dependencies

1. Open an R console and install the following libraries from a nearby mirror.
   `install.packages(c('dplyr', 'DBI', 'RSQLite'))`

## Database setup

A pre-initialized SQLite database is located under `src/resources/sql/results.sqlite3`.
Alternatively, you can setup your own SQLite database and use the schema `src/resources/sql/results_schema.sql`.
Other RDMS might require minor modifications of the schema.

## Data

If the used datasets are not present under `data/raw`, follow these steps.

### Test data

#### Citysearch

1. Download from `https://www.research.cs.rutgers.edu/~gganu/` 
   or directly from `https://www.research.cs.rutgers.edu/~gganu/datasets/ManuallyAnnotated_Corpus.txt`.
2. Place this dataset at `data/raw/test/citysearch_test.xml`

#### SemEval Laptop

1. Download the [SemEval 2015 Laptop](https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools) dataset 
   from  `http://metashare.ilsp.gr:8080/repository/browse/semeval-2015-absa-laptops-reviews-test-data-gold-annotations/a2bd9f229ca111e4a350842b2b6a04d7d9091e92fc7149f485037cb9e98809af/` 
   and `http://metashare.ilsp.gr:8080/repository/download/4ab77724612011e4acce842b2b6a04d73cf3cb586f894d30b3c8afdd98cfbdc8/`
2. Unpack the archives and rename to XML files to `semeval_lapt_2015_test.xml` and `semeval_lapt_2015_train.xml`
3. Move the files to `data/raw/test/`

#### Other datasets

The code is written to support the SemEval 2014, 2015, 2016 Laptop and Restaurant datasets as well as a Foursquare dataset.
By default, these datasets are not processed or evaluated. Minor changes in the source code have to be made to do so.

### Training data

#### Amazon Reviews

1. Get the Amazon Electronics dataset from [here](http://jmcauley.ucsd.edu/data/amazon/links.html)

* Electronics 5-core (0.4 GB): <http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz>
* Metadata (3.1 GB): <http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz>

1. Unzip these files and rename them to `amazon_reviews_electronics.json` and `amazon_metadata.json`.
2. Place the JSON files in `data/raw/train`. 

#### Yelp Reviews

1. Get the Yelp dataset (4.9 GB): <https://www.yelp.com/dataset/download>
2. Extract `yelp_academic_dataset_business.json` and `yelp_academic_dataset_business.json`
   to `data/raw/train`. 

## Pre-processing

Set the Python Path to `src` . On Unix-based environments this can be done by running `export PYTHONPATH=src` from the repository root. This also applies to any other Python Script in this repository. These scripts should be executed from the repository root. E.g. `python src/data/train/process_amazon.py` .

Run the pre-processing scripts.
These scripts expect raw input data in `data/raw/train` and output pre-processed data in `data/train`.

* `src/data/train/process_amazon.py`
* `src/data/train/process_yelp.py`
* `src/data/test/parse_citysearch.py`
* `src/data/test/parse_semeval_datasets.py`

## Training

### Word2Vec

Train Word2Vec models using `src/preprocessing/create_embeddings.py`. 
Multiple Word2Vec models are outputted to `data/embeddings`.
To reproduce our results set the environment variable PYTHONHASHSEED=42 before running Python.
On UNIX-based environments this can be done using `export PYTHONHASHSEED=42`.
Training of Word2Vec is performed single-threaded to allow reproducibility.
The parameter `workers=1` can be increased to make use of multiple CPU cores.     

### Aspect candidates

Extract nouns from the training corpora using `src/preprocessing/create_noun_count.py`.
This will output lists of nouns sorted by frequency to `data/noun_count`. 

## Experiments

The experiments are started by `src/experiments/experiment.py`.
The classified sentences are written to the table `results` of the database. 

## Evaluation

Evaluation of the results is done in R.
The script `src/evalution/evaluation.R` prints the tables to standard out and writes the plots in PDF format to `out/plots`.

# Hints

Since this is not a production environment, there are only few safe guards for nonexistent files, sane input, null, etc.

Files are read into memory and then processed.
Thus, at least 10 GB of RAM is required to process the Yelp dataset.
One could modify the preprocessing scripts to operate on the files line by line.
While this requires very little memory, this approach much slower.
