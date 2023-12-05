import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Example using Seaborn's built-in dataset
df = sns.load_dataset("iris")

# print (df.columns)
# print(df.shape)
# print(df.index)
# print (df.describe().T)

# Display the first few rows of the dataset
# print(df.head())

# # Summary statistics
# print(df.describe())

# # Check for missing values
# print(df.isnull().sum())

# # Example: Pairplot for multivariate analysis
# sns.pairplot(df, hue="species")
# plt.show()

# # Example: Correlation Heatmap
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show()



# Example: Perform one-way ANOVA on sepal length for the three species
setosa_sepal_length = df[df['species'] == 'setosa']['sepal_length']
versicolor_sepal_length = df[df['species'] == 'versicolor']['sepal_length']
virginica_sepal_length = df[df['species'] == 'virginica']['sepal_length']

# Perform ANOVA
statistic, p_value = f_oneway(setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length)

# Print results
print(f"ANOVA Statistic: {statistic}")
print(f"P-Value: {p_value}")

# Compare p-value to significance level (e.g., 0.05) for decision
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in sepal length among the species.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in sepal length among the species.")

# Combine the sepal lengths and species into a new DataFrame
sepal_data = pd.DataFrame({
    'Species': np.repeat(['Setosa', 'Versicolor', 'Virginica'], [len(setosa_sepal_length), len(versicolor_sepal_length), len(virginica_sepal_length)]),
    'Sepal Length': np.concatenate([setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length])
})

# Create a bar plot
sns.barplot(x='Species', y='Sepal Length', data=sepal_data, ci='sd')  # 'sd' for standard deviation
plt.title('Mean Sepal Lengths by Species')
plt.show()

# Create a swarm plot
sns.swarmplot(x='Species', y='Sepal Length', data=sepal_data)
plt.title('Distribution of Sepal Lengths by Species')
plt.show()