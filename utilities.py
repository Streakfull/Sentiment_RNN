import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

# Some constants
word_to_int = {}
int_to_word = {}
vocab_size = 0
sequence_length = 200


def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def adjust_review_length(review):
    if(len(review) > sequence_length):
        return review[0: sequence_length]
    zero_array = np.zeros(sequence_length-len(review))
    return np.concatenate((zero_array, review))


def initialize_dictionaries(reviews_frame):
    global vocab_size, word_to_int, int_to_word
    all_words_list = np.array([element.lower().split()
                               for element in reviews_frame.values.flatten()])
    unique_words = sorted(set(np.hstack(all_words_list).tolist()))
    vocab_size = len(unique_words) + 1
    word_to_int = {word: index+1 for index, word in enumerate(unique_words)}
    int_to_word = {word: integer for integer, word in word_to_int.items()}


# converting a sentence to a list of encodings from word_to_int dictionary
def encode_sentence(sentence):
    encoded_sentence = [word_to_int[word]
                        for word in sentence.split()]
    return encoded_sentence


def get_optimizer(optimizer_name, paramaters, learning_rate, weight_decay=0):
    if optimizer_name == "Adam":
        return optim.Adam(paramaters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "SGD":
        return optim.SGD(paramaters, lr=learning_rate)
    return None


def get_loss_function(function_name):
    if function_name == "BCELoss":
        return nn.BCELoss()


def get_vocab_size():
    return vocab_size


def calculate_accuracy(predicted, true_results):
    sum = 0
    for i, prediction in enumerate(predicted):
        if round(prediction.item()) == true_results[i].item():
            sum += 1
    return sum/len(predicted)


def log_trial(params_dict, training_accuracy="N/A", validation_accuracy="N/A", test_accuracy="N/A"):
    trial_log = open("trialLog.txt", "a")
    trial_log.write("Training accuracy: {}".format(training_accuracy))
    trial_log.write("\n")
    trial_log.write("Validation accuracy: {}".format(validation_accuracy))
    trial_log.write("\n")
    trial_log.write("Test accuracy: {}".format(test_accuracy))
    trial_log.write("\n")
    trial_log.write(str(params_dict))
    trial_log.write("\n")
    trial_log.write("\n")
    trial_log.close()
