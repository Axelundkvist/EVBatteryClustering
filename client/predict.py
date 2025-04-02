import os
import sys
import torch
from data import load_data
from model import load_parameters
from fedn.utils.helpers.helpers import save_metrics
import traceback

# Set directory path for safe module import
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

# Ensure FEDN_DATA_PATH is set in environment
data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")


def predict(in_model_path, out_json_path, data_path):
    """Predict SoH using the trained model.

    :param in_model_path: Path to the input model file.
    :param out_json_path: Path to output prediction JSON file.
    :param data_path: Path to input data file for testing.
    """
    # Load test data
    x_test, y_test, recent_stats = load_data(data_path, is_train=False)

    # Ensure x_test is a tensor of correct type
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)  # Optional, if you need ground truth

    print(f"[DEBUG] x_test shape: {x_test.shape}")
    print(f"[DEBUG] y_test shape: {y_test.shape} (Ground truth, not used for prediction)")

    # Load trained model
    model = load_parameters(in_model_path)
    model.eval()  # Set to evaluation mode

    # Perform prediction
    with torch.no_grad():
        y_pred = model(x_test)  # Regression output (continuous values)

    # Prepare and save result
    result = {"predicted_SoH": y_pred.squeeze().tolist()}  # Squeeze to flatten for easier output

    print(f"[DEBUG] Predictions made, sample: {result['predicted_SoH'][:5]}")  # Optional: preview first 5
    save_metrics(result, out_json_path)
    print(f"[INFO] Predictions saved to {out_json_path}")


if __name__ == "__main__":
    print("\n\n[DEBUG] ===== Starting prediction() via __main__ =====\n\n")
    if len(sys.argv) < 3:
        print("[ERROR] Not enough arguments provided to predict.py")
        print("Usage: python predict.py <in_model_path> <out_json_path>")
        sys.exit(1)
    try:
        predict(sys.argv[1], sys.argv[2], data_path)
    except Exception as e:
        print("\n\n[ERROR] Something went wrong in main predict.py execution!\n")
        traceback.print_exc()
