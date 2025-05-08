import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


DATASET_PATH = 'diabetes_health_indicators_dataset.csv' 

TARGET_COLUMN = 'Diabetes_binary' # Target variable for prediction


HIDDEN_UNITS_L1 = 128
HIDDEN_UNITS_L2 = 64
OUTPUT_UNITS = 1 
LEARNING_RATE = 0.001
EPOCHS = 50 
BATCH_SIZE = 64
TEST_SIZE = 0.2 
RANDOM_STATE = 42 

# Paths for saving the model and preprocessor
SAVED_MODEL_PATH = 'diabetes_mlp_model.pth'
SAVED_PREPROCESSOR_PATH = 'preprocessor.joblib'


class DiabetesPredictorMLP(nn.Module):
    def __init__(self, input_features, hidden_l1, hidden_l2, output_features):
        super(DiabetesPredictorMLP, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_features, hidden_l1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # Dropout for regularization

        # Layer 2
        self.fc2 = nn.Linear(hidden_l1, hidden_l2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Output Layer
        self.fc3 = nn.Linear(hidden_l2, output_features)
        self.sigmoid = nn.Sigmoid() # Sigmoid for binary classification probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x) # Output a probability between 0 and 1
        return x

# --- 2. Load and Prepare Data ---
def load_and_preprocess_data(file_path, target_column):
    """
    Loads the dataset, preprocesses it, and splits it into training and testing sets.
    Returns processed data, labels, and the fitted preprocessor.
    """
    global INPUT_FEATURES # To update the global variable for model instantiation

    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset '{file_path}' loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check the path.")
        return None, None, None, None, None

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        available_columns = df.columns.tolist()
        print(f"Available columns: {available_columns}")
        # Attempt to find a common alternative if 'Diabetes_binary' is missing
        if 'Diabetes_012_health_indicators_BRFSS2015.csv' in file_path and 'Diabetes_012' in available_columns:
            print(f"Found 'Diabetes_012'. If this is your target, update TARGET_COLUMN.")
            print("Note: 'Diabetes_012' is for multiclass. This script is for binary (e.g., 'Diabetes_binary').")
        return None, None, None, None, None

    # Define features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column].astype(int) # Ensure target is integer

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Target value counts:\n{y.value_counts()}")

    # Identify numerical features (all features in this dataset are expected to be numeric)
    numerical_features = X.columns.tolist()
    if not numerical_features:
        print("Error: No numerical features identified. Check your dataset.")
        return None, None, None, None, None
    print(f"\nIdentified {len(numerical_features)} numerical features for scaling: {numerical_features}")


    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough' # In case any columns were missed (should not happen here)
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )

    # Fit the preprocessor on the training data and transform both training and testing sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Update INPUT_FEATURES based on the processed data's shape
    INPUT_FEATURES = X_train_processed.shape[1]
    print(f"Number of input features after preprocessing: {INPUT_FEATURES}")

    print(f"\nShape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

# --- 3. Training Function ---
def train_model_pytorch(model, train_loader, criterion, optimizer, epochs, device):
    """
    Trains the PyTorch neural network.
    """
    model.train() # Set the model to training mode
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
    print("--- Training Finished ---")

# --- 4. Evaluation Function ---
def evaluate_model_pytorch(model, test_loader, criterion, device):
    """
    Evaluates the PyTorch model on the test set.
    """
    model.eval() # Set the model to evaluation mode
    test_loss = 0.0
    all_labels = []
    all_predictions = []

    print("\n--- Starting Evaluation ---")
    with torch.no_grad(): # Disable gradient calculations
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)

            predicted_probs = outputs.cpu().numpy()
            predicted_labels = (predicted_probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
            all_predictions.extend(predicted_labels.flatten())

    avg_test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Ensure target names match your binary classes (0 and 1)
    print(classification_report(all_labels, all_predictions, target_names=['No Diabetes (0)', 'Diabetes/Prediabetes (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    print("--- Evaluation Finished ---")
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data
    X_train_p, X_test_p, y_train_arr, y_test_arr, fitted_preprocessor = load_and_preprocess_data(DATASET_PATH, TARGET_COLUMN)

    if X_train_p is None or fitted_preprocessor is None:
        print("Exiting due to data loading/preprocessing error.")
    else:
        # Save the fitted preprocessor
        joblib.dump(fitted_preprocessor, SAVED_PREPROCESSOR_PATH)
        print(f"\nPreprocessor saved to '{SAVED_PREPROCESSOR_PATH}'")

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_p, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_p, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_arr, dtype=torch.float32)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"\nDataLoaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        # 2. Initialize the model (INPUT_FEATURES is set in load_and_preprocess_data)
        pytorch_model = DiabetesPredictorMLP(
            input_features=INPUT_FEATURES, # This global variable is updated by load_and_preprocess_data
            hidden_l1=HIDDEN_UNITS_L1,
            hidden_l2=HIDDEN_UNITS_L2,
            output_features=OUTPUT_UNITS
        ).to(device)
        print(f"\nPyTorch Model initialized:\n{pytorch_model}")

        # 3. Define loss function and optimizer
        criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
        optimizer = optim.Adam(pytorch_model.parameters(), lr=LEARNING_RATE)
        print(f"Optimizer: Adam, Learning Rate: {LEARNING_RATE}")
        print(f"Loss Function: BCELoss")

        # 4. Train the model
        train_model_pytorch(pytorch_model, train_loader, criterion, optimizer, EPOCHS, device)

        # 5. Evaluate the model
        test_accuracy = evaluate_model_pytorch(pytorch_model, test_loader, criterion, device)

        # 6. Save the trained PyTorch model
        torch.save(pytorch_model.state_dict(), SAVED_MODEL_PATH)
        print(f"\nTrained PyTorch model state saved to '{SAVED_MODEL_PATH}'")

        print("\n--- Script Finished ---")
