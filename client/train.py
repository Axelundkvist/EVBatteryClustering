import math
import os
import sys

import torch
import torch.optim as optim
from model import load_parameters, save_parameters
from data import load_data, estimate_soh, StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from data import load_data
from fedn.utils.helpers.helpers import save_metadata

TEMP_THRESHOLD = 40  # Define later based on observations


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def train(in_model_path, out_model_path, data_path):
    """ Train the SoH model only if the temperature threshold is exceeded. """

    # Load and preprocess data
    X_train, y_train = load_data(data_path)

    # Check if training should proceed (based on recent 14-day average temperature)
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    last_14_days = df.sort_values("timestamp", ascending=False).head(14)
    mean_temp = last_14_days["batMaxTemp"].mean()

    if mean_temp <= TEMP_THRESHOLD:
        print(f"Temperature threshold not exceeded ({mean_temp:.2f}°C). Skipping training.")
        return  # Stop execution

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Load model
    model = load_parameters(in_model_path)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save trained model
    save_parameters(model, out_model_path)


'''
def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    # Load data
    x_train, y_train = load_data(data_path)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    n_batches = int(math.ceil(len(x_train) / batch_size))
    criterion = torch.nn.NLLLoss()
    for e in range(epochs):  # epoch loop
        for b in range(n_batches):  # batch loop
            # Retrieve current batch
            batch_x = x_train[b * batch_size : (b + 1) * batch_size]
            batch_y = y_train[b * batch_size : (b + 1) * batch_size]
            # Train on batch
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # Log
            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
'''