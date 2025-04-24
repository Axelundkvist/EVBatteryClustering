import collections
import os

import torch

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


## model class name should be BatterySoHModel

class BatterySoHModel(torch.nn.Module):
    def __init__(self, input_dim=21):  # Updated to 21 features to match data
        super(BatterySoHModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def compile_model(num_features=21):  # Updated default to 21
    """ Create a fresh model instance with the correct input size. """
    return BatterySoHModel(input_dim=num_features)  # Ensure input size matches dataset


def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path, num_features):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :param num_features: Number of input features for the model
    :type num_features: int
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    print(f"[DEBUG] Creating model with {num_features} input features")
    model = compile_model(num_features)
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file {model_path} does not exist. Using a new model.")
        return model
    
    try:
        parameters_np = helper.load(model_path)
        
        # Check if the parameters match the model architecture
        expected_params = len(model.state_dict().keys())
        if len(parameters_np) != expected_params:
            print(f"[WARNING] Parameter count mismatch. Expected {expected_params}, got {len(parameters_np)}")
            print("[WARNING] Using a new model instead.")
            return model
        
        params_dict = zip(model.state_dict().keys(), parameters_np)
        state_dict = collections.OrderedDict({key: torch.tensor(x, dtype=torch.float32) for key, x in params_dict})
        
        # Try to load the state dict, but catch any size mismatch errors
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[DEBUG] Successfully loaded model parameters")
        except RuntimeError as e:
            print(f"[WARNING] Error loading state dict: {e}")
            print("[WARNING] Using a new model instead.")
            return model
            
        return model
    except Exception as e:
        print(f"[WARNING] Error loading parameters: {e}")
        print("[WARNING] Using a new model instead.")
        return model
    

def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model(num_features=21)
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
