import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Map target numbers to species names
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_mapping)

# Choose a continuous variable to visualize
variable_to_visualize = 'petal length (cm)'

# Create a histogram for the continuous variable
plt.figure(figsize=(10, 6))
sns.histplot(iris_df[variable_to_visualize], kde=True, bins=30)
plt.title(f'Distribution of {variable_to_visualize}')
plt.xlabel(variable_to_visualize)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a bar chart for the categorical variable
plt.figure(figsize=(10, 6))
sns.countplot(x='species', data=iris_df, palette='viridis')
plt.title('Distribution of Species in the Iris Dataset')
plt.xlabel('Species')
plt.ylabel('Count')
plt.grid(True)
plt.show()
