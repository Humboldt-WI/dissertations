import mariadb
from itertools import repeat
from tqdm.contrib.itertools import product
from cat.simple import get_scores
from reach import Reach
from experiments.helpers.attention_functions import rbf_attention
from experiments.helpers.results_repository import ResultsRepository

N_ASPECT_CANDIDATES = 200


def cat_predict(model, aspect_candidates, labels, sentences, gamma=0.03,
                attention_func=rbf_attention):
    """Call CAt to predict labels for a list of sentences.
    Set of labels and aspect candidates is passed to CAt."""
    s = get_scores(sentences,
                   aspect_candidates,
                   model,
                   labels,
                   gamma=gamma,
                   attention_func=attention_func)

    y_pred = s.argmax(1)
    return y_pred


def encodings_to_labels(y_list, labels):
    """Resolve list of encoded label values."""
    return [labels[x] for x in y_list]


def zip_sentences_predictions(y_true, y_pred, sentences, labels):
    """Create a list of true and predicted labels."""
    y_true_labeled = encodings_to_labels(y_true, labels)
    y_pred_labeled = encodings_to_labels(y_pred, labels)

    sentences_joined = [' '.join(s) for s in sentences]

    zipped = list(zip(sentences_joined, y_true_labeled, y_pred_labeled))
    return zipped


def get_aspect_candidates(name, n=N_ASPECT_CANDIDATES):
    """Get n most frequent nouns by corpus name. """
    noun_path = "data/noun_count/" + name + ".txt"

    with open(noun_path, 'r') as noun_file:
        nouns = noun_file.read().splitlines()

    # Cut sorted list at n. This returns the n most frequent nouns.
    nouns = nouns[:n]

    # cat expects input to be [[x_1], [x_2], ..., [x_n]]
    nouns = [[noun] for noun in nouns]
    return nouns


def get_embeddings(name, sample_size=100, i=0):
    """Get embeddings by corpus name."""
    w2v_model_path = "data/embeddings/" + name + "-" + str(sample_size) + "-" + str(i) + ".w2v"
    model = Reach.load(w2v_model_path, unk_word="<UNK>")
    return model


repo = ResultsRepository()


def do_experiment(pairs, iterations=None, corpus_sample_rates=None, gammas=None, attentions=None):
    if corpus_sample_rates is None:
        corpus_sample_rates = [100]

    if iterations is None:
        iterations = [0]

    if gammas is None:
        gammas = [0.03]

    if attentions is None:
        attentions = [rbf_attention]

    # Loop over Cartesian product of all parameters
    p = product(pairs, corpus_sample_rates, iterations, gammas, attentions)

    for (train_dataset_name, test_dataset), corpus_sample_rate, iteration, gamma, attention in p:
        test_dataset_name = test_dataset.__name__
        attention_name = attention.__name__

        model = get_embeddings(train_dataset_name, corpus_sample_rate, iteration)
        aspect_candidates = get_aspect_candidates(train_dataset_name)

        # Get sentences and labels for test data set
        sentences, y_true, labels = test_dataset()

        # Get predicted labels for dataset
        y_pred = cat_predict(
            model=model,
            sentences=sentences,
            labels=labels,
            aspect_candidates=aspect_candidates,
            gamma=gamma,
            attention_func=attention
        )

        # Create a list of tuples with sentence, y_true, y_pred
        sentences_labeled = zip_sentences_predictions(y_true, y_pred, sentences, labels)

        # Create a list with parameters for each labeled sentence in the dataset
        parameters = (
            test_dataset_name, train_dataset_name, iteration, corpus_sample_rate, gamma,
            attention_name)
        zipped = zip(repeat(parameters), sentences_labeled)
        sentences_params_list = [left + right for left, right in zipped]

        repo.bulk_insert(sentences_params_list)
