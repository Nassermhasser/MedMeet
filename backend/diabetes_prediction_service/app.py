import flask
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib # For loading the preprocessor


HIDDEN_UNITS_L1 = 128
HIDDEN_UNITS_L2 = 64
OUTPUT_UNITS = 1
MODEL_PATH = 'diabetes_mlp_model.pth' 
PREPROCESSOR_PATH = 'preprocessor.joblib' 

class DiabetesPredictorMLP(nn.Module):
    def __init__(self, input_features, hidden_l1, hidden_l2, output_features):
        super(DiabetesPredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_l1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_l1, hidden_l2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_l2, output_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Preprocessor '{PREPROCESSOR_PATH}' loaded successfully.")

    if hasattr(preprocessor, 'transformers_') and preprocessor.transformers_:

        if hasattr(preprocessor, 'n_features_in_'):
             INPUT_FEATURES = preprocessor.n_features_in_
        elif hasattr(preprocessor, 'transformers_') and len(preprocessor.transformers_) > 0 and \
             hasattr(preprocessor.transformers_[0][1], 'n_features_in_'): # Check inside the first transformer pipeline
             INPUT_FEATURES = preprocessor.transformers_[0][1].n_features_in_
        else:

            expected_features = [
                'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
            ]
            INPUT_FEATURES = len(expected_features)
            print(f"Warning: Could not automatically determine INPUT_FEATURES from preprocessor. Using fallback value: {INPUT_FEATURES}")
    else:

        expected_features = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        INPUT_FEATURES = len(expected_features)
        print(f"Warning: Could not automatically determine INPUT_FEATURES from preprocessor. Using fallback value: {INPUT_FEATURES}")

    print(f"Determined/Fallback INPUT_FEATURES: {INPUT_FEATURES}")

except FileNotFoundError:
    print(f"Error: Preprocessor file '{PREPROCESSOR_PATH}' not found. The application will not work correctly.")
    preprocessor = None
    INPUT_FEATURES = 21 
    # Exit if preprocessor is essential and not found
    # exit()

# Load model

if preprocessor: # Only load model if preprocessor was loaded
    model = DiabetesPredictorMLP(INPUT_FEATURES, HIDDEN_UNITS_L1, HIDDEN_UNITS_L2, OUTPUT_UNITS)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. The application will not work correctly.")
        model = None # Set model to None if loading failed
        # exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure INPUT_FEATURES, HIDDEN_UNITS_L1, etc., match the saved model's architecture.")
        model = None
        # exit()
else:
    model = None # No preprocessor, so no model operations

# --- 3. Initialize Flask App ---
app = flask.Flask(__name__)

# --- 4. Define Prediction Endpoint ---
@app.route('/')
def home():

    return "Diabetes Prediction API is running. Use the /predict endpoint with POST request."

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return flask.jsonify({'error': 'Model or preprocessor not loaded. Check server logs.'}), 500

    try:
        # Get data from POST request
        data = flask.request.get_json(force=True)
        print(f"Received data: {data}")


        feature_names = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        
        # Create a DataFrame from the input data, ensuring correct column order
        # Convert all incoming data to numeric, as the model expects numeric inputs
        input_data_dict = {}
        for feature in feature_names:
            try:
                input_data_dict[feature] = [float(data[feature])] # Convert to float, handle potential errors
            except KeyError:
                return flask.jsonify({'error': f'Missing feature: {feature}'}), 400
            except ValueError:
                return flask.jsonify({'error': f'Invalid value for feature: {feature}. Must be numeric.'}), 400

        input_df = pd.DataFrame.from_dict(input_data_dict)
        print(f"Input DataFrame:\n{input_df}")

        # Preprocess the data
        processed_data = preprocessor.transform(input_df)
        print(f"Processed data shape: {processed_data.shape}")

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad(): # Disable gradient calculations
            prediction_output = model(input_tensor)
            prediction_prob = prediction_output.item() # Get single probability value

        # Convert probability to class (0 or 1)
        predicted_class = 1 if prediction_prob > 0.5 else 0
        prediction_label = "Diabetes/Prediabetes" if predicted_class == 1 else "No Diabetes"

        # Return prediction as JSON
        return flask.jsonify({
            'prediction_probability': round(prediction_prob, 4),
            'predicted_class': predicted_class,
            'prediction_label': prediction_label
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return flask.jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- 5. Run the App ---
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
