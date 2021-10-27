# Extract notebook reviews from the Amazon consumer electronics dataset.
# Reviews are tokenized, converted to lower case and written to a single corpus.

# This can be sped up by using basic shell commands to pre-filter the reviews.
# grep -F "[['Electronics', 'Computers & Accessories', 'Laptops']]" amazon_metadata.json | cut -d "'" -f4 > laptop_asin.txt
# grep -f laptop_asin.txt amazon_metadata.json > amazon_metadata_pre_grepped.json
# grep -f laptop_asin.txt amazon_reviews_electronics.json > amazon_reviews_electronics_pre_grepped.json

import ast
import json
from tqdm import tqdm
from data.helper_functions import strip_lower_tokenize

# Input, output files
amazon_metadata_in_path = "data/raw/train/amazon_metadata.json"
amazon_electronics_reviews_in_path = "data/raw/train/amazon_reviews_electronics.json"
amazon_laptop_reviews_out_path = "data/train/amazon_laptop.txt"

# Extract all ASINs (= Amazon Standard Identification Number) from the metadata file
# that belong to the laptop subcategory.
amazon_laptop_subcategory = "[['Electronics', 'Computers & Accessories', 'Laptops']]"
amazon_laptop_asins = []

with open(amazon_metadata_in_path, "r") as infile:
    lines = infile.readlines()
    for line in tqdm(lines):
        # Simple string comparison instead of loading the JSON file as this a cheaper operation.
        if amazon_laptop_subcategory in line:
            # Use ast.literal_eval instead of json.loads because of single quote usage
            product = ast.literal_eval(line)

            asin = product['asin']
            amazon_laptop_asins.append(asin)

# Use the ASINs the extract laptop reviews from the dataset.
with open(amazon_laptop_reviews_out_path, "w") as outfile, \
        open(amazon_electronics_reviews_in_path, "r") as infile:
    lines = infile.readlines()

    # Each line of the file is a JSON.
    # Iterate over each review and check whether the ASIN is contained in the list of
    # laptop ASINs.
    for line in tqdm(lines):
        line_json = json.loads(line)

        asin = line_json['asin']
        if asin in amazon_laptop_asins:
            text = line_json['reviewText']

            # tokenize the review and append to output file
            tokenized_string = strip_lower_tokenize(text)
            outfile.write(tokenized_string + "\n")
