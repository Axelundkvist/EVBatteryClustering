import collections

import torch

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


## model class name should be BatterySoHModel

class BatterySoHModel(torch.nn.Module):
    # if you're going to start using the US dataset, change the input_dim to 8 or the number of features 
    def __init__(self, input_dim=6):  # 6 features in dataset
        super(BatterySoHModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def compile_model():
    """ Create a fresh model instance with the correct input size. """
    return BatterySoHModel(input_dim=6)  # Ensure input size matches dataset


def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x, dtype=torch.float32) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model
    

def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
