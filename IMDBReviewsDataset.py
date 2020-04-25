from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.tensor as tensor
import numpy as np
import torch
from utilities import get_device, word_to_int, int_to_word, initialize_dictionaries, encode_sentence, adjust_review_length, word_to_int
from string import punctuation

np.random.seed(0)
torch.manual_seed(0)


class IMDBReviewsDataset(Dataset):

    def __init__(self):
        self.labels_frame = pd.read_csv("./dataIMDB/labels.txt")  # (25000,1)
        convert_labels(self.labels_frame)
        self.reviews_frame = pd.read_csv("./dataIMDB/reviews.txt")  # (25000,1)
        self.reviews_frame.columns = ['review']
        self.reviews_frame = sanitize_reviews(self.reviews_frame)
        initialize_dictionaries(self.reviews_frame)
        # each review is encoded and of length seq_length
        encode_adjust_length(self.reviews_frame)

    def __len__(self):
        return self.labels_frame.shape[0]

    def __getitem__(self, index):
        review = tensor(
            self.reviews_frame.iloc[index, 0], dtype=torch.int64)
        label = tensor(
            self.labels_frame.iloc[index, 0], dtype=torch.int64)
        return {'review': review, 'label': label}


# Adjusts labels to binary numbers


def convert_labels(labels_frame):
    label_map = {"positive": 1.0, "negative": 0.0}
    labels_frame['label'] = labels_frame['label'].map(label_map)


# Encodes all reviews then sets all reviews to a specified sequence length by padding or truncating
def encode_adjust_length(reviews_frame):
    reviews = reviews_frame.values.flatten()
    for index, review in enumerate(reviews):
        reviews[index] = adjust_review_length(encode_sentence(review))
    reviews_frame["review"] = reviews


def sanitize_reviews(review_frame):
    review_frame['review'] = review_frame['review'].str.replace(
        '[{}]'.format(punctuation), '')
    review_frame['review'] = review_frame['review'].str.replace(
        "br", '', regex=False)
    return review_frame


def get_data_sets():
    imdb_reviews = IMDBReviewsDataset()
    training_data_length = round(0.8 * imdb_reviews.__len__())
    validation_data_length = round(
        (imdb_reviews.__len__() - training_data_length)/2)
    training_data, validation_data, test_data = random_split(
        imdb_reviews, (training_data_length, validation_data_length, validation_data_length))
    #      20,000          2500             2500
    return training_data, validation_data, test_data
