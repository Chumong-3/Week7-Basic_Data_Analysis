import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from seaborn
    df = sns.load_dataset('iris')
    
    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check for missing values
    print("\nMissing values in dataset:")
    print(df.isnull().sum())

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nDataset Summary Statistics:")
print(df.describe())

# Group by species and calculate mean petal length
species_means = df.groupby('species')['petal_length'].mean()
print("\nAverage Petal Length per Species:")
print(species_means)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# 1️⃣ Line Chart - Trends over Time (Using index for illustration)
plt.subplot(2, 2, 1)
plt.plot(df.index, df['sepal_length'], label='Sepal Length', color='b')
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.legend()

# 2️⃣ Bar Chart - Average petal length per species
plt.subplot(2, 2, 2)
species_means.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')

# 3️⃣ Histogram - Distribution of Sepal Width
plt.subplot(2, 2, 3)
sns.histplot(df['sepal_width'], bins=15, kde=True, color='purple')
plt.title('Distribution of Sepal Width')

# 4️⃣ Scatter Plot - Sepal Length vs Petal Length
plt.subplot(2, 2, 4)
sns.scatterplot(x=df['sepal_length'], y=df['petal_length'], hue=df['species'])
plt.title('Sepal Length vs Petal Length')

plt.tight_layout()
plt.show()
