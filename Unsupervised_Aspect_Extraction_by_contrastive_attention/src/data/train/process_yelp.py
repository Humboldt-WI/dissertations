import json
import random
from tqdm import tqdm
from data.helper_functions import strip_lower_tokenize

# Set seed to make results reproducible
random.seed(42)

# Input, Output files
yelp_metadata_path = "data/raw/train/yelp_academic_dataset_business.json"
yelp_reviews_path = "data/raw/train/yelp_academic_dataset_review.json"
yelp_restaurant_reviews_out_path = "data/train/yelp_restaurant.txt"

# Get list of restaurant IDs
restaurant_ids = []
with open(yelp_metadata_path) as infile:
    # This can be done line by line if memory is an issue. Loading everything into memory is faster.
    lines = infile.readlines()
    for line in tqdm(lines):
        business = json.loads(line)
        try:
            if "Restaurants" in business['categories']:
                restaurant_id = business['business_id']
                restaurant_ids.append(restaurant_id)
        except TypeError:
            # some businesses have categories = None.
            # This throws a TypeError. Just catch and ignore it
            pass

# Sample size
n_samples = 50_000

# This can be done more memory efficient if you know the distribution of business types,
# then you don't need to load the full dataset into memory.
with open(yelp_restaurant_reviews_out_path, "w") as outfile, \
        open(yelp_reviews_path, "r") as infile, \
        tqdm(total=n_samples) as pbar:
    lines = infile.readlines()

    # Randomly shuffle the list of lines
    random.shuffle(lines)

    n = 0
    # loop over each line until n reaches sample size
    for line in lines:
        review = json.loads(line)
        review_business_id = review['business_id']
        if review_business_id in restaurant_ids:
            text = review['text']
            text_tokenized = strip_lower_tokenize(text)
            outfile.write(text_tokenized + "\n")

            pbar.update(1)
            n += 1

            # Break out of loop as soon as 50k samples are reached
            if n == n_samples:
                break
