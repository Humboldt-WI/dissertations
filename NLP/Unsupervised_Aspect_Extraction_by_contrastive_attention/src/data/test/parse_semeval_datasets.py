# Script to extract sentences with a single label from SemEval datasets.
# SemEval 2014, 2015, 2016 and Foursquare are supported.
# Sentences and labels are written to separate files, as the CAt implementation
# by Tulkens/Van Cranenburgh expects this.

from lxml import etree
from data.helper_functions import strip_lower_tokenize

TEST_DATA_PATH = 'data/test/'
TEST_DATA_RAW_PATH = 'data/raw/test/'


def extract_and_split_semeval(xml_in: str,
                              sentence_out_file: str,
                              labels_out_file: str,
                              opinions_xml_node: str,
                              opinion_xml_node: str) -> None:
    """Extract sentences and their label with a distinct label from the dataset."""
    root = etree.parse(xml_in)

    labels = []
    sentences = []

    # Loop over the XML nodes using XPath.
    # Aspect labels are either aspectCategories/aspectCategory or Opinions/Opinion.
    # The resulting XPath is either './/sentence[count(aspectCategories/aspectCategory' or
    # './/sentence[count(Opinions/Opinion)'.
    # Only sentences with exactly one distinct label are considered.
    for s in root.xpath(
            './/sentence[count(' + opinions_xml_node + '/' + opinion_xml_node + ') > 0]'):
        sentence_labels = s.xpath('./' + opinions_xml_node + '/' + opinion_xml_node + '/@category')

        # Split labels over '#' thus removing sub categories
        sentence_labels = [label.split("#")[0] for label in sentence_labels]

        # Sentences with multiple different labels (e.g. food + service) are discarded
        # Sentences with multiple same labels are kept (e.g. food + food) are kept
        if len(set(sentence_labels)) != 1:
            continue

        sentence = s.find('text').text
        label = sentence_labels[0].lower()

        # Tokenize the sentence.
        sentence_tokenized = strip_lower_tokenize(sentence)

        labels.append(label)
        sentences.append(sentence_tokenized)

    # Write both lists to file
    with open(sentence_out_file, 'w') as text_out:
        for sentence in sentences:
            text_out.write("%s\n" % sentence)

    with open(labels_out_file, 'w') as labels_out:
        for label in labels:
            labels_out.write("%s\n" % label)


def extract_and_split_semeval_wrapper(dataset: str, opinions_xml_node: str,
                                      opinion_xml_node: str) -> None:
    """Wrapper function to handle filenames."""
    xml_in = TEST_DATA_RAW_PATH + dataset + ".xml"
    sentence_out_file = TEST_DATA_PATH + dataset + ".txt"
    labels_out_file = TEST_DATA_PATH + dataset + "_labels.txt"

    extract_and_split_semeval(xml_in=xml_in,
                              sentence_out_file=sentence_out_file,
                              labels_out_file=labels_out_file,
                              opinions_xml_node=opinions_xml_node,
                              opinion_xml_node=opinion_xml_node
                              )


def concatenate_files(infile_paths: list[str], outfile_path: str) -> None:
    """Concatenate a list of files to a single file"""
    with open(outfile_path, 'w') as outfile:
        for infile_path in infile_paths:
            with open(infile_path, 'r') as infile:
                outfile.write(infile.read())


# Uncomment relevant lines to use further datasets.
# extract_and_split_semeval_wrapper("semeval_rest_2014_test", "aspectCategories", "aspectCategory")
# extract_and_split_semeval_wrapper("semeval_rest_2015_test", "Opinions", "Opinion")
# extract_and_split_semeval_wrapper("semeval_rest_2016_test", "Opinions", "Opinion")
extract_and_split_semeval_wrapper("semeval_lapt_2015_test", "Opinions", "Opinion")
extract_and_split_semeval_wrapper("semeval_lapt_2015_train", "Opinions", "Opinion")
# extract_and_split_semeval_wrapper("semeval_lapt_2016_test", "Opinions", "Opinion")
# extract_and_split_semeval_wrapper("foursquare", "Opinions", "Opinion")


# Concatenate SemEval Laptop 2015 test and train

# Dictionary of files to datasets to concatenate. combined file -> list of files to concatenate
# Only used for SemEval Laptop 2015 train and test dataset, however can be reused with other
# datasets if required. In that case, just at the mapping to this dictionary.

testsets_to_concat = {
    # concatenated file basename -> [list of testset basenames]
    "semeval_lapt_2015_combined": ["semeval_lapt_2015_test", "semeval_lapt_2015_train"]
}

# For each testset, concatenate sentence file and label file
for (combined_test_name, testsets) in testsets_to_concat.items():
    combined_testset_sent_path = TEST_DATA_PATH + combined_test_name + ".txt"
    testset_sent_paths = [TEST_DATA_PATH + t + ".txt" for t in testsets]
    concatenate_files(testset_sent_paths, combined_testset_sent_path)

    combined_testset_label_path = TEST_DATA_PATH + combined_test_name + "_labels.txt"
    testset_label_paths = [TEST_DATA_PATH + t + "_labels.txt" for t in testsets]
    concatenate_files(testset_label_paths, combined_testset_label_path)
