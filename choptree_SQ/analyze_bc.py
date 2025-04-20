import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_bc import PolicyNetwork 
from sklearn.metrics import mean_squared_error, r2_score

# Load processed data
def load_data(filename="processed_data/processed_good.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    print("Loading data and model...")
    data = load_data()

    observations = np.array(data['observations'], dtype=np.float32)
    actions = np.array(data['actions'], dtype=np.float32)

    # Load model
    input_dim = observations.shape[1]
    output_dim = actions.shape[1]
    model = PolicyNetwork(input_dim, output_dim)
    model.load_state_dict(torch.load("models/bc_model.pth"))
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(observations)).numpy()

    # --- Metrics ---
    mse = mean_squared_error(actions, predictions)
    r2 = r2_score(actions, predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(actions.flatten(), label="Actual", alpha=0.7)
    plt.plot(predictions.flatten(), label="Predicted", alpha=0.7)
    plt.title("Predicted vs Actual Actions")
    plt.legend()

    plt.subplot(1, 2, 2)
    errors = np.abs(actions - predictions)
    plt.hist(errors.flatten(), bins=50)
    plt.title("Action Error Distribution")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
