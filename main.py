# Import necessary libraries
from sklearn import tree
import pandas as pd

# Sample dataset: Features of different species (e.g., size, color, habitat)
# Replace this with your actual data
data = {
    'Feature1': [5.1, 7.0, 6.3, 1.2],
    'Feature2': [3.5, 3.2, 3.3, 5.5],
    'Feature3': [9.1, 4.0, 7.3, 1.4],
    'Feature4': [0.2, 1.4, 2.5, 0.5],
    'Species': ['Species A', 'Species B', 'Species C', 'Species D']
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Define the features and the target variable
features = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']]
target = df['Species']

# Initialize the decision tree classifier
classifier = tree.DecisionTreeClassifier()

# Train the classifier with the dataset
classifier.fit(features, target)

# Function to predict the species
def predict_species(new_features):
    # The new_features should be a list of features
    prediction = classifier.predict([new_features])
    return prediction[0]

# Example usage of the function
# Replace the list with actual features to get the species prediction
example_features = [5.1, 3.5, 9.1, 0.2]  # Example feature set
print('The predicted species is: ',predict_species(example_features))
