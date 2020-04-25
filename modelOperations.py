import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utilities import get_device, get_vocab_size, get_optimizer, get_loss_function, calculate_accuracy
from RNNmodel import RNNmodel
import time
from IMDBReviewsDataset import get_data_sets
import torch
import numpy as np
import matplotlib.pyplot as plt


def perform_forward_loop(model, batch, device):
    hidden_state, cell_state = model.init_hidden(len(batch['review']))
    reviews = batch['review'].long().to(device)
    labels = batch['label'].long().to(device)
    output = model(reviews, (hidden_state.to(
        device), cell_state.to(device)))
    return output, labels


def plot(losses):
    epochs = range(1, len(losses)+1)
    plt.plot(epochs, losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def train(training_data, model, loss_function, learning_rate, epochs, device, batch_size=512,  optimizer="Adam", weight_decay=0):
    model.to(device)
    model.train()
    start_time = time.time()
    optimizer = get_optimizer(
        "Adam", model.parameters(), learning_rate, weight_decay)
    criterion = get_loss_function(loss_function)
    data_loader = DataLoader(training_data, batch_size=batch_size)
    all_loss = np.zeros((epochs,))
    for epoch in range(epochs):
        for _, batch in enumerate(data_loader):
            optimizer.zero_grad()
            output, labels = perform_forward_loop(model, batch, device)
            loss = criterion(output.squeeze().float(), labels.float())
            all_loss[epoch] += loss
            loss.backward()
            optimizer.step()
        # Printing stats
        print('Epoch %d loss is %f' % (epoch, float(all_loss[epoch])))
    total_time = time.time() - start_time
    # plot(all_loss)
    if device == 'cuda:0':
        torch.cuda.empty_cache()
    print("TIME:", total_time)


def evaluate(data, model, device):
    with torch.no_grad():
        data_loader = DataLoader(
            data, batch_size=512)
        model.to(device)
        model.eval()
        total_accuracy = 0
        for index, batch in enumerate(data_loader):
            predicted, labels = perform_forward_loop(model, batch, device)
            total_accuracy += calculate_accuracy(predicted, labels)
        return total_accuracy/(index+1)
