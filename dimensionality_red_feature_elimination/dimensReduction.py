from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('training90.csv')

d = {'Unsatisfactory': 0, 'Below Average': 1, 'Average': 2, 'Above Average': 3, 'Excellent': 4}
data['Rating'] = data['Rating'].map(d)


# Instantiate the PCA object with the number of components
pca = PCA(n_components=2)

# Fit the model on the data
pca.fit(data)

# Transform the data into 2D
data_transformed = pca.transform(data)

print(data.shape)
print(data_transformed.shape)
print("reduced the attribute number from 11 to 2\n")

plt.figure(figsize=(8,6))
plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['Rating'], cmap='plasma')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
