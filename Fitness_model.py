import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset from CSV
print("Loading dataset from CSV...")
df = pd.read_csv("Fitness_Data.csv")

# Display basic statistics
print(f"\nDataset loaded with {len(df)} rows and {len(df.columns)} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nTraining Intensity Distribution:")
print(df['Training_Intensity'].value_counts())

# Prepare data for machine learning
print("\nPreparing data for machine learning...")

# Create label encoders for categorical variables
le_fitness = LabelEncoder()
le_disabilities = LabelEncoder()
le_goal = LabelEncoder()
le_target = LabelEncoder()

# Encode categorical variables
df_encoded = df.copy()
df_encoded['Fitness_Level_encoded'] = le_fitness.fit_transform(df['Fitness_Level'])
df_encoded['Disabilities_encoded'] = le_disabilities.fit_transform(df['Disabilities'])
df_encoded['Fitness_Goal_encoded'] = le_goal.fit_transform(df['Fitness_Goal'])
df_encoded['Training_Intensity_encoded'] = le_target.fit_transform(df['Training_Intensity'])

# Prepare features and target
feature_columns = ['Age', 'Weight', 'Height', 'Muscle_Mass', 'BMI', 
                  'Fitness_Level_encoded', 'Disabilities_encoded', 'Fitness_Goal_encoded']
X = df_encoded[feature_columns]
y = df_encoded['Training_Intensity_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree model
print("Training Decision Tree model...")
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    criterion='gini'
)

dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Training Complete!")
print(f"Model Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
target_names = le_target.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Client input and prediction function
def predict_training_intensity():
    print("\n" + "="*50)
    print("FITNESS TRAINING INTENSITY PREDICTOR")
    print("="*50)
    
    try:
        # Get user inputs
        print("\nPlease enter the following client information:")
        
        age = float(input("Age: "))
        weight = float(input("Weight (kg): "))
        height = float(input("Height (cm): "))
        muscle_mass = float(input("Muscle Mass (%): "))
        
        print("\nFitness Level options: high, moderate, low")
        fitness_level = input("Fitness Level: ").lower().strip()
        while fitness_level not in ['high', 'moderate', 'low']:
            fitness_level = input("Please enter 'high', 'moderate', or 'low': ").lower().strip()
        
        print("\nDisabilities options: Yes, No")
        disabilities = input("Disabilities (Yes/No): ").strip()
        while disabilities.lower() not in ['yes', 'no']:
            disabilities = input("Please enter 'Yes' or 'No': ").strip()
        disabilities = 'Yes' if disabilities.lower() == 'yes' else 'No'
        
        bmi = float(input("BMI: "))
        
        print("\nFitness Goal options: Muscle Gain, Weight Loss")
        fitness_goal = input("Fitness Goal: ").strip()
        while fitness_goal not in ['Muscle Gain', 'Weight Loss']:
            fitness_goal = input("Please enter 'Muscle Gain' or 'Weight Loss': ").strip()
        
        # Encode the inputs
        fitness_level_encoded = le_fitness.transform([fitness_level])[0]
        disabilities_encoded = le_disabilities.transform([disabilities])[0]
        fitness_goal_encoded = le_goal.transform([fitness_goal])[0]
        
        # Create feature array
        features = np.array([[age, weight, height, muscle_mass, bmi, 
                            fitness_level_encoded, disabilities_encoded, fitness_goal_encoded]])
        
        # Make prediction
        prediction_encoded = dt_model.predict(features)[0]
        prediction = le_target.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability
        prediction_proba = dt_model.predict_proba(features)[0]
        confidence = max(prediction_proba) * 100
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Recommended Training Intensity: {prediction}")
        print(f"Confidence: {confidence:.1f}%")
        print("="*50)
        
        return prediction
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure all inputs are valid.")
        return None

# Run the prediction system
predict_training_intensity()
