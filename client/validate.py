import os
import sys
import torch
from model import load_parameters
from data import load_data
from fedn.utils.helpers.helpers import save_metrics
import traceback
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

# Load FEDN_DATA_PATH from environment variable
data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")

print(f"[DEBUG] validate.py running with sys.argv: {sys.argv}")


def convert_to_tensor(data, dtype=torch.float32):
    """Ensure data is converted to a PyTorch tensor, handling Pandas and NumPy cases correctly."""
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return torch.tensor(data.to_numpy(), dtype=dtype)
    elif isinstance(data, np.ndarray):  # NumPy arrays don't need .to_numpy()
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, torch.Tensor):  # Already a tensor
        return data
    else:
        raise TypeError(f"Unsupported data type for conversion: {type(data)}")

def validate(in_model_path, out_json_path, data_path):
    """Validate model for SoH regression.

    :param in_model_path: Path to the input model file.
    :param out_json_path: Path to output JSON file.
    :param data_path: Path to data file.
    """
    print("[DEBUG] Loading data for validation...")

    # Load data
    x_train, y_train, recent_stats = load_data(data_path)
    x_test, y_test, recent_stats = load_data(data_path, is_train=False)


    # ✅ Apply this function for all conversions
    x_train = convert_to_tensor(x_train, dtype=torch.float32)
    y_train = convert_to_tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test = convert_to_tensor(x_test, dtype=torch.float32)
    y_test = convert_to_tensor(y_test, dtype=torch.float32).view(-1, 1)


    print("[DEBUG] Data successfully loaded and converted to tensors.")
    print(f"[DEBUG] Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    # Load model
    model = load_parameters(in_model_path)
    model.eval()  # Set to evaluation mode
    print("[DEBUG] Model loaded and set to eval mode.")

    # ✅ Criterion for regression
    criterion = torch.nn.MSELoss()

    # Evaluate
    with torch.no_grad():
        train_out = model(x_train)
        training_loss = criterion(train_out, y_train).item()
        print(f"[DEBUG] Training Loss: {training_loss:.4f}")

        test_out = model(x_test)
        test_loss = criterion(test_out, y_test).item()
        print(f"[DEBUG] Test Loss: {test_loss:.4f}")

    # Save metrics
    result = {
        "training_loss": training_loss,
        "test_loss": test_loss
    }
    save_metrics(result, out_json_path)
    print(f"[INFO] Validation results saved to {out_json_path}")


if __name__ == "__main__":
    print("\n\n[DEBUG] ===== Starting validate() via __main__ =====\n\n")
    if len(sys.argv) < 3:
        print("[ERROR] Not enough arguments provided to validate.py")
        print("Usage: python validate.py <in_model_path> <out_json_path>")
        sys.exit(1)
    try:
        validate(sys.argv[1], sys.argv[2], data_path)
    except Exception as e:
        print("\n\n[ERROR] Something went wrong in main validate.py execution!\n")
        traceback.print_exc()

