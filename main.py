import torch
from IMDBReviewsDataset import get_data_sets
from utilities import get_vocab_size, get_device, log_trial
from modelOperations import train, evaluate
from RNNmodel import RNNmodel


def main():
    # Loading all data sets
    training_data, validation_data, test_data = get_data_sets()
    device = get_device()
    accuracy_threshold = 0.86

    # model_params
    layers = 1
    vocab_size = get_vocab_size()
    embedding_dim = 400
    hidden_size = 256
    lstm_dropout = 0
    layer_dropout = 0

    # training_params
    loss_function = "BCELoss"
    optimizer = "Adam"
    batch_size = 512
    epochs = 4
    weight_decay = 0
    learning_rate = 1e-3

    # Model params
    # Training params
    model = RNNmodel(vocab_size, embedding_dim, hidden_size,
                     lstm_dropout, layer_dropout, layers)
    # To load a pre-trained model
    # model.load_state_dict(torch.load("./state"))

    train(training_data, model, loss_function, learning_rate,
          epochs, device, batch_size, optimizer, weight_decay)
    # TODO: Add more evaluation metrics (precision,recall)

    training_accuracy = evaluate(training_data, model, device)
    # TODO: Use an F1 score between precision,recall of both training and validation sets
    if(training_accuracy > accuracy_threshold):
        torch.save(model.state_dict(), "./state")

    validation_accuracy = evaluate(validation_data, model, device)

    print("TRAINING ACCURACY", training_accuracy)
    print("VALIDATION ACCURACY", validation_accuracy)

    #test_accuracy = evaluate(test_data, model, device)

    params_dict = {
        "model_params": {
            "layers": layers,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "lstm_dropout": lstm_dropout,
            "layer_dropout": layer_dropout
        },
        "training_params": {
            "loss_function": loss_function,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": weight_decay
        }
    }

    log_trial(params_dict, training_accuracy,
              validation_accuracy)


main()
