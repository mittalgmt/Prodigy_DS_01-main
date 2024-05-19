import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Choose a continuous variable to visualize
variable_to_visualize = 'petal length (cm)'

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(iris_df[variable_to_visualize], kde=True, bins=30)
plt.title(f'Distribution of {variable_to_visualize}')
plt.xlabel(variable_to_visualize)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
