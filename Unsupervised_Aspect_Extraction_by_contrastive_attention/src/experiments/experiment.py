# This script runs all experiments in the thesis.

from experiments.helpers.experiment_handler import do_experiment
import experiments.helpers.testset_loaders as ts
import experiments.helpers.attention_functions as att

# Pairs of training corpus and test dataset.
# Unused datasets are commented out.
pairs = [
    # train, test
    ("yelp_restaurant", ts.citysearch),
    # ("yelp_restaurant", ts.semeval_rest_2014_test),
    # ("yelp_restaurant", ts.semeval_rest_2015_test),
    # ("yelp_restaurant", ts.semeval_rest_2016_test),
    # ("yelp_restaurant", ts.foursquare),
    # ("amazon_laptop", ts.semeval_lapt_2015_train),
    # ("amazon_laptop", ts.semeval_lapt_2015_test),
    ("amazon_laptop", ts.semeval_lapt_2015_combined),
    # ("amazon_laptop", ts.semeval_lapt_2016_test),
]

# Main experiment
do_experiment(pairs)

# Sub-experiment corpus size
# Sample corpus to 5 %, 10 %, ..., 100 %
# Repeat 10 times
corpus_sample_rates = range(5, 105, 5)
iterations = range(0, 10)

do_experiment(pairs, iterations=iterations, corpus_sample_rates=corpus_sample_rates)

# Sub-experiment influence of gamma
# Gamma set to 0, 0.01, 0.02, ..., 0.05
# RBF kernel as well as mean weighting are used for the attention mechanism
gammas = [x / 1000 for x in range(0, 51, 1)]
attentions = [att.rbf_attention, att.no_attention]

do_experiment(pairs, gammas=gammas, attentions=attentions)
