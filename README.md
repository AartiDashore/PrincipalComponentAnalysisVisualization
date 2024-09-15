# Principal Component Analysis (PCA) Visualization

## Introduction
Principal Component Analysis (PCA) is a widely used dimensionality reduction technique that transforms data into a set of linearly uncorrelated variables known as principal components. These components capture the most variance in the data. This project visualizes PCA results in both 2D and 3D, enabling an easy understanding of how PCA reduces dimensions while preserving important data characteristics.

## Features
- **PCA Dimensionality Reduction**: Reduces the dataset to 2 or 3 dimensions using PCA.
- **Visualization**: 2D and 3D scatter plots to visualize the transformed data.
- **Explained Variance**: Displays how much variance each principal component captures.

## Requirements
Before running the code, ensure you have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required libraries by running:

```bash
pip install numpy matplotlib scikit-learn
```

## Code Structure

### 1. **Data Loading**
```python
from sklearn.datasets import load_iris

# Load dataset (using Iris dataset as an example)
data = load_iris()
X = data.data
y = data.target
```
- **Concept**: We load the Iris dataset, which consists of 150 samples from three species of flowers. Each sample has four features (sepal length, sepal width, petal length, and petal width).
- **Objective**: `X` is the feature matrix (data points), and `y` is the target variable representing class labels (species).

### 2. **Applying PCA**
```python
from sklearn.decomposition import PCA

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance
```
- **Concept**: PCA transforms the dataset into a new set of coordinates called principal components. The number of components can be specified (here, 2 or 3).
  - **Eigenvalues and Eigenvectors**: PCA computes eigenvalues and eigenvectors of the covariance matrix to identify the principal components.
  - **Explained Variance**: The `explained_variance_ratio_` tells us how much of the data's total variance is captured by each principal component.
- **Objective**: The function returns the transformed data `X_pca` and the proportion of variance explained by each component.

### 3. **2D Scatter Plot Visualization**
```python
import matplotlib.pyplot as plt

def plot_pca_2d(X_pca, y, explained_variance):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title(f'2D PCA (Explained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    plt.show()
```
- **Concept**: In 2D PCA visualization, the first two principal components are plotted on the x and y axes. These components capture the most variance in the dataset.
- **Color Mapping**: The data points are color-coded based on their class labels (`y`), using the `viridis` colormap for differentiation.
- **Objective**: We visualize the transformed dataset in 2D, showing how PCA compresses the data while preserving most of the variance.

### 4. **3D Scatter Plot Visualization**
```python
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_3d(X_pca, y, explained_variance):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
    ax.set_title(f'3D PCA (Explained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}, {explained_variance[2]:.2f})')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    fig.colorbar(scatter)
    plt.show()
```
- **Concept**: In 3D PCA, the first three principal components are used to create a 3D scatter plot.
  - **Axes**: Principal Component 1 (PC1) on the x-axis, Principal Component 2 (PC2) on the y-axis, and Principal Component 3 (PC3) on the z-axis.
- **Objective**: This visualization gives a more comprehensive view of the dataset by showing the first three principal components.

### 5. **Executing the Code**
```python
# Run PCA for 2 components
X_pca_2d, explained_variance_2d = apply_pca(X, n_components=2)
plot_pca_2d(X_pca_2d, y, explained_variance_2d)

# Run PCA for 3 components
X_pca_3d, explained_variance_3d = apply_pca(X, n_components=3)
plot_pca_3d(X_pca_3d, y, explained_variance_3d)
```
- **Concept**: 
  - First, PCA is applied to reduce the dataset to 2 components, and a 2D scatter plot is generated.
  - Then, PCA is applied for 3 components, and a 3D scatter plot is generated.
- **Objective**: This step runs the PCA on the dataset and visualizes the results in both 2D and 3D.

## How to Run the Code

1. **Download or Clone the Project**: You can either copy the code or clone this repository.
   
2. **Install the Required Libraries**:
   Install the necessary Python packages by running:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Run the Script**:
   Simply execute the Python script. The code will display two plots:
   - A 2D PCA scatter plot with the explained variance for each principal component.
   - A 3D PCA scatter plot with the explained variance for the top three principal components.

4. **Modify for Your Dataset**: 
   Replace the Iris dataset (`X` and `y` variables) with your dataset if needed. Ensure that the data is preprocessed (numerical and standardized) before applying PCA.

## Concepts Used

- **Principal Component Analysis (PCA)**: PCA is a technique that reduces the dimensionality of data while preserving as much variance as possible. It does so by transforming the original variables into a new set of variables (principal components), which are uncorrelated and ordered by the amount of variance they capture.
  
- **Dimensionality Reduction**: PCA helps reduce the number of features in a dataset, which simplifies models and helps visualize high-dimensional data in 2D or 3D.

- **Eigenvalues and Eigenvectors**: PCA relies on finding the eigenvectors (principal components) and eigenvalues (explained variance) of the covariance matrix. The eigenvectors represent the directions of maximum variance, while the eigenvalues tell how much variance is captured along each eigenvector.

- **Scatter Plot**: A visualization technique used to display the relationship between variables. In this project, we use scatter plots to show the relationship between the principal components in 2D and 3D.

## Output
PCA Output 2D:

![PCA_Output_2D](https://github.com/AartiDashore/PrincipalComponentAnalysisVisualization/blob/main/PCA_Output_1.png)

PCA Output 3D:

![PCA_Output_3D](https://github.com/AartiDashore/PrincipalComponentAnalysisVisualization/blob/main/PCA_Output_2.png)

## Conclusion

This project provides a simple and intuitive way to apply and visualize PCA for dimensionality reduction. The 2D and 3D scatter plots allow us to observe how well the reduced dimensions capture the important characteristics of the dataset.
