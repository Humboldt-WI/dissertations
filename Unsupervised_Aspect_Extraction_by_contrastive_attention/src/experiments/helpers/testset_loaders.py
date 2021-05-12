# Loaders for annotated datasets. The loader provided by Tulkens/Van Cranenburgh ist used.

from cat.dataset import loader


def citysearch():
    """Load the Citysearch dataset."""
    return loader(
        instance_path="data/test/citysearch.txt",
        label_path="data/test/citysearch_labels.txt",
        subset_labels=[
            "ambience",
            "staff",
            "food"
        ])


def semeval_rest_2014_test():
    """Load the SemEval 2014 Restaurant (test) dataset."""
    return loader(
        instance_path="data/test/semeval_rest_2014_test.txt",
        label_path="data/test/semeval_rest_2014_test_labels.txt",
        subset_labels=[
            "ambience",
            "service",
            "food"
        ])


def semeval_rest_2015_test():
    """Load the SemEval 2015 Restaurant (test) dataset."""
    return loader(
        instance_path="data/test/semeval_rest_2015_test.txt",
        label_path="data/test/semeval_rest_2015_test_labels.txt",
        subset_labels=[
            "ambience",
            "service",
            "food"
        ])


def semeval_rest_2016_test():
    """Load the SemEval 2016 Restaurant (test) dataset."""
    return loader(
            instance_path="data/test/semeval_rest_2016_test.txt",
            label_path="data/test/semeval_rest_2016_test_labels.txt",
            subset_labels=[
                "ambience",
                "service",
                "food"
            ])


def foursquare():
    """Load the Foursquare dataset."""
    return loader(
            instance_path="data/test/foursquare.txt",
            label_path="data/test/foursquare_labels.txt",
            subset_labels=[
                "ambience",
                "service",
                "food"
            ])


def semeval_lapt_2015_test():
    """Load the SemEval 2015 Laptop (test) dataset."""
    return loader(
            instance_path="data/test/semeval_lapt_2015_test.txt",
            label_path="data/test/semeval_lapt_2015_test_labels.txt",
            subset_labels=[
                "os",
                "keyboard",
                "battery",
                "company",
                "support",
            ])


def semeval_lapt_2016_test():
    """Load the SemEval 2016 Laptop (train) dataset."""
    return loader(
        instance_path="data/test/semeval_lapt_2015_test.txt",
        label_path="data/test/semeval_lapt_2015_test_labels.txt",
        subset_labels=[
            "os",
            "keyboard",
            "battery",
            "company",
            "support",
        ])

def semeval_lapt_2016_test():
    """Load the SemEval 2016 Laptop (test) dataset."""
    return loader(
        instance_path="data/test/semeval_lapt_2016_test.txt",
        label_path="data/test/semeval_lapt_2016_test_labels.txt",
        subset_labels=[
            "battery",
            "os",
            "keyboard",
            "battery",
            "company",
            "support",
        ])


def semeval_lapt_2015_combined():
    """Load the SemEval 2015 Laptop (combined test+train) dataset."""
    return loader(
        instance_path="data/test/semeval_lapt_2015_combined.txt",
        label_path="data/test/semeval_lapt_2015_combined_labels.txt",
        subset_labels=[
            "support",
            "battery",
            "display",
            "mouse",
            "software",
            "keyboard",
            "company",
            "os",
            "graphics"
        ])
