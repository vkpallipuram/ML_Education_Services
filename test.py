from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit the PCA model to the dataset
iris_pca = pca.fit_transform(iris.data)

# Plot the loadings as a heatmap
sns.heatmap(pca.components_.T, cmap='coolwarm', annot=True)

# Plot the transformed data
sns.scatterplot(x=iris_pca[:,0], y=iris_pca[:,1], hue=iris.target)
# sns.xlabel('PC1')
# sns.ylabel('PC2')
sns.show()
