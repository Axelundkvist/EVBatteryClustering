import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict
import collections
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

# =============================================================================
# Utility Functions
# =============================================================================

def init_seed(seed_file: str):
    """
    Initialize seeds for reproducibility.
    Expects seed_file to be a .npz file with a key 'seed'.
    """
    seeds = np.load(seed_file)
    seed_value = int(seeds['seed'])
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seed initialized to {seed_value} from {seed_file}.")

def compile_model() -> nn.Module:
    """
    Instantiate and compile the BatterySOHEstimator model.
    """
    model = BatterySOHEstimator()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model compiled and moved to {device}.")
    return model

def save_parameters(model: nn.Module, file_path: str):
    """
    Save model parameters to file using the helper module.
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, file_path)
    print(f"Model parameters saved to {file_path}.")

def load_parameters(model: nn.Module, file_path: str, public_keys: List[str] = None):
    """
    Load model parameters from file using the helper module.
    If public_keys is provided, update only those keys,
    leaving other parameters (private parameters) unchanged.
    """
    parameters_np = helper.load(file_path)
    current_state = model.state_dict()
    
    # Convert numpy parameters back to tensors
    params_dict = zip(current_state.keys(), parameters_np)
    loaded_state = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    
    if public_keys is not None:
        # Update only keys in public_keys
        for key in loaded_state:
            if key in public_keys and key in current_state:
                current_state[key] = loaded_state[key]
        model.load_state_dict(current_state)
        print(f"Loaded parameters for public keys from {file_path}.")
    else:
        model.load_state_dict(loaded_state)
        print(f"Loaded all parameters from {file_path}.")

# =============================================================================
# Battery SOH Estimator Model
# =============================================================================

class BatterySOHEstimator(nn.Module):
    def __init__(self, electrical_dim: int = 4, physical_dim: int = 2,
                 hidden_electrical: int = 16, hidden_physical: int = 8,
                 combined_hidden: int = 16):
        """
        Neural network that estimates battery State of Health (SOH) using a dual-branch architecture:
          - Electrical branch processes 4 electrical features (V_mean, V_std, V_kur, V_sk)
          - Physical branch processes 2 physical features (T_mean, S_mean)
        The branches are then fused to produce a single scalar output.
        """
        super(BatterySOHEstimator, self).__init__()
        
        # Electrical branch for 4 features
        self.electrical_branch = nn.Sequential(
            nn.Linear(electrical_dim, hidden_electrical),
            nn.ReLU(),
            nn.Linear(hidden_electrical, hidden_electrical),
            nn.ReLU()
        )
        
        # Physical branch for 2 features
        self.physical_branch = nn.Sequential(
            nn.Linear(physical_dim, hidden_physical),
            nn.ReLU(),
            nn.Linear(hidden_physical, hidden_physical),
            nn.ReLU()
        )
        
        # Combined branch to fuse the two branches and produce SOH estimate
        self.combined_branch = nn.Sequential(
            nn.Linear(hidden_electrical + hidden_physical, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, 1)  # Output: SOH (a scalar value)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x shape: [batch_size, 6] (first 4 electrical, last 2 physical)
        Output: 1D tensor with SOH prediction for each sample.
        """
        # Split input into two branches
        x_electrical = x[:, :4]
        x_physical   = x[:, 4:]
        
        # Process each branch
        out_elec = self.electrical_branch(x_electrical)
        out_phys = self.physical_branch(x_physical)
        
        # Concatenate branch outputs and compute final prediction
        combined = torch.cat((out_elec, out_phys), dim=1)
        soh = self.combined_branch(combined)
        return soh.squeeze(1)  # Remove extra dimension

# =============================================================================
# Main: For testing purposes only
# =============================================================================

if __name__ == "__main__":
    init_seed("../seed.npz")
    
    # Example: compile the model and test on dummy data.
    
    #device="cpu"
    model = compile_model()
    
    # Create a dummy batch: 10 samples, 6 features each.
    #dummy_input = torch.randn(10, 6)
    #predictions = model(dummy_input)
    
