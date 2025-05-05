import inspect
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import time


import fedn.network.grpc.fedn_pb2 as fedn
from fedn.network.combiner.hooks.hooks import FunctionServiceServicer
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase
from fedn.network.combiner.modelservice import bytesIO_request_generator, model_as_bytesIO
#from client.train import train, train_linear_soh_model, load_parameters, save_metadata

from client.data import load_data

from torch.utils.data import TensorDataset, DataLoader

from client.model import compile_model


def validate_model(model, X_val, y_val):
    model.eval()
    criterion = torch.nn.MSELoss()

    # 1) unwrap pandas if needed
    if isinstance(X_val, (pd.DataFrame, pd.Series)):
        X_arr = X_val.values
    else:
        X_arr = X_val
    if isinstance(y_val, (pd.DataFrame, pd.Series)):
        y_arr = y_val.values
    else:
        y_arr = y_val
        
    if X_val.ndim == 1:
        X_val = X_val.reshape(-1, 1)

    # 2) to pure NumPy float32
    X_arr = np.array(X_arr, dtype=np.float32)
    y_arr = np.array(y_arr, dtype=np.float32).reshape(-1, 1)

    # ─── Now it’s safe to check for NaNs ─────────────────────────────
    # print("  ▶ X_arr NaN?", np.isnan(X_arr).any(),
    #       "y_arr NaN?", np.isnan(y_arr).any())
    # ─────────────────────────────────────────────────────────────────

    # 3) finally to torch.Tensor
    X_t = torch.from_numpy(X_arr)
    y_t = torch.from_numpy(y_arr)

    with torch.no_grad():
        output = model(X_t)
        loss   = criterion(output, y_t)
    return loss.item()



def train_aifcl_model(model, X_train, y_train, batch_size=10, epochs=5, lr=0.01):
    model.train()
    #print("  ▶ X_train NaN?", np.isnan(X_train).any(),"y_train NaN?", np.isnan(y_train).any())
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
    except:
        print("fel med xtrain")
    try:
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values
    except:
        print("fel med ytrain")
    
    #X_train = np.array(X_train)
    #y_train = np.array(y_train)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    data = TensorDataset(X_train, y_train)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    return model

def train_model(model, X_train, y_train, batch_size=10, epochs=5, lr=0.01):
    model.train()
    #print("  ▶ X_train NaN?", np.isnan(X_train).any(),"y_train NaN?", np.isnan(y_train).any())
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
    except:
        print("fel med xtrain")
    try:
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values
    except:
        print("fel med ytrain")
    
    #X_train = np.array(X_train)
    #y_train = np.array(y_train)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    
    data = TensorDataset(X_train, y_train)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for e in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()  # last batch loss

    return model, final_loss



def vanilla_train_model(model, X_train, y_train, batch_size=10, epochs=5, lr=0.01):
    model.train()
    #print("  ▶ X_train NaN?", np.isnan(X_train).any(),"y_train NaN?", np.isnan(y_train).any())
    # if it’s a pandas object, pull out its underlying array
    
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
    except:
        print("fel med xtrain")
    try:
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values
    except:
        print("fel med ytrain")
    
    #X_train = np.array(X_train)
    #y_train = np.array(y_train)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    y_train = y_train.reshape(-1, 1)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    
    data = TensorDataset(X_train, y_train)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = None

    for e in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

    return model, final_loss

def export(obj, path='aifcl_log.json'):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)



class Client_Sim:
    def __init__(self, id: str, dist: int):
        self.id: str = id
        self.dist: int = dist
        self.X: torch.tensor = None
        self.y: torch.tensor = None
        self.loss_history: List = []
        self.stats = None
    
    def gen_data(self, n_samples=100, input_dim=8, filepath= None, current_round = 0, total_rounds = 5):
        # This setup (very different distributions) produces clusters fantastically!
        # torch.manual_seed(int(time.time()) + int(self.id))  # deterministic per client
        #torch.manual_seed(int(self.dist))

        # TODO: så att denna funktionen generarar ditt fleet data set
        try:
            #print(f"succesfully loaded data from {filepath}")
            self.X, self.y, self.stats = load_data(filepath, round_id=current_round, total_rounds=total_rounds)
        except:
            print(f"ERROR : loading data from load_data failed")
            print()
            mean = self.dist * 30.0  # very large gap between clients
            std = 0.5                # low variance to keep data tight

            self.X = torch.randn(n_samples, input_dim) * std + mean
            self.y = torch.full((n_samples,), self.dist, dtype=torch.float32)  # label = client ID

            self.y = self.y.view(-1, 1)     
        
        
        '''
        path for the nasaFiltered dataset, for columns check your obsidian file or at the end of cleaning.ipynb
        '''
        #/Users/Axel/fedn/examples/server-functions/new_data




    def act(self, config, input_dim=8, round = 0):
        mode = config['mode']

        input_dim = self.X.shape[1]        
        model = compile_model(num_features=input_dim)
        

        if mode != "log":
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), config['model'])}
            model.load_state_dict(state_dict)


        if mode == "train":
            model, loss = train_model(model, self.X, self.y)
            self.loss_history.append(loss)
            return model, loss
        elif mode == "train_aifcl":
            model = train_aifcl_model(model, self.X, self.y)
            return model, None
        elif mode == "validate":
            loss = validate_model(model, self.X, self.y)
            self.loss_history.append(loss)
            return model, loss
        elif mode == "vanilla":
            # print(type(self.X))
            # print(type(self.y))
            model, loss = vanilla_train_model(model, self.X, self.y)
            #print("från act funktionen")
            #print(f"model: {model}")
            print(f'Vanilla strikes!')
            val_loss = validate_model(model, self.X, self.y)
            print(f'val loss = {val_loss}')
            return model, loss, val_loss
        elif mode == "log":
            export_dict = {k: v for k, v in config.items() if k != 'model'}
            export(export_dict)
            return model, None
        else:
            raise ValueError(f"Unknown mode: {mode}")