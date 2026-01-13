"""
PCA Implementation from Scratch
================================

A complete implementation of Principal Component Analysis using only NumPy.
This implementation follows the sklearn API for easy comparison and understanding.

Author: Agricultural ML Learning Project
Purpose: Educational - Understanding PCA internals
"""

import numpy as np
from typing import Optional, Union


class PCA:
    """
    Principal Component Analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, float or None, optional (default=None)
        Number of components to keep.
        - If int: number of components
        - If float (0 < n_components < 1): select number of components such that
          the variance retained is greater than the percentage specified
        - If None: keep all components

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (eigenvectors).

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to eigenvalues of the covariance matrix.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    n_components_ : int
        The estimated number of components (after fitting).

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.

    Examples
    --------
    >>> import numpy as np
    >>> from pca_implementation import PCA
    >>>
    >>> # Create sample data
    >>> X = np.array([[2.5, 2.4],
    ...               [0.5, 0.7],
    ...               [2.2, 2.9],
    ...               [1.9, 2.2],
    ...               [3.1, 3.0]])
    >>>
    >>> # Fit PCA
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    >>>
    >>> # Transform data
    >>> X_transformed = pca.transform(X)
    >>>
    >>> # Get explained variance
    >>> print(pca.explained_variance_ratio_)
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None):
        """
        Initialize PCA.

        Parameters
        ----------
        n_components : int, float or None
            Number of components to keep.
        """
        self.n_components = n_components

        # Attributes to be set during fit
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the PCA model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Store data dimensions
        self.n_samples_, self.n_features_ = X.shape

        # Step 1: Center the data (subtract mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute covariance matrix
        # Cov = (X^T * X) / (n - 1)
        cov_matrix = np.cov(X_centered.T)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Determine number of components to keep
        if self.n_components is None:
            # Keep all components
            n_components = self.n_features_
        elif isinstance(self.n_components, int):
            # Keep specified number of components
            n_components = min(self.n_components, self.n_features_)
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Keep components that explain specified variance ratio
            cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_components = np.argmax(cumulative_variance_ratio >= self.n_components) + 1
        else:
            raise ValueError(
                f"n_components={self.n_components} must be an int in [1, n_features], "
                f"a float in (0, 1), or None"
            )

        self.n_components_ = n_components

        # Step 6: Select top n_components
        self.components_ = eigenvectors[:, :n_components].T  # Transpose for consistency
        self.explained_variance_ = eigenvalues[:n_components]

        # Step 7: Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Check if model is fitted
        if self.components_ is None:
            raise ValueError("PCA model is not fitted yet. Call 'fit' first.")

        # Center the data
        X_centered = X - self.mean_

        # Project onto principal components
        # X_transformed = X_centered * components^T
        X_transformed = X_centered.dot(self.components_.T)

        return X_transformed

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the model with X and apply dimensionality reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        X_transformed : ndarray of shape (n_samples, n_components)
            Data in the transformed (PCA) space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data transformed back to original space.
        """
        # Check if model is fitted
        if self.components_ is None:
            raise ValueError("PCA model is not fitted yet. Call 'fit' first.")

        # Project back to original space
        # X_reconstructed = X_transformed * components + mean
        X_reconstructed = X_transformed.dot(self.components_) + self.mean_

        return X_reconstructed

    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.

        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance of data.
        """
        if self.components_ is None:
            raise ValueError("PCA model is not fitted yet. Call 'fit' first.")

        # Covariance = components^T * diag(explained_variance) * components
        components_full = self.components_.T
        cov = components_full.dot(
            np.diag(self.explained_variance_)
        ).dot(components_full.T)

        return cov

    def get_precision(self) -> np.ndarray:
        """
        Compute data precision matrix (inverse covariance).

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Estimated precision of data.
        """
        if self.components_ is None:
            raise ValueError("PCA model is not fitted yet. Call 'fit' first.")

        cov = self.get_covariance()
        precision = np.linalg.inv(cov)

        return precision

    def score(self, X: np.ndarray, y=None) -> float:
        """
        Return the average log-likelihood of all samples.

        This is a simple reconstruction error metric (negative MSE).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to score.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        score : float
            Negative mean squared reconstruction error.
        """
        # Transform and reconstruct
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)

        # Calculate reconstruction error (MSE)
        mse = np.mean((X - X_reconstructed) ** 2)

        # Return negative MSE (higher is better)
        return -mse

    def __repr__(self) -> str:
        """String representation of the PCA object."""
        if self.components_ is None:
            return f"PCA(n_components={self.n_components}) - Not fitted"
        else:
            return (
                f"PCA(n_components={self.n_components_})\n"
                f"  Explained variance: {self.explained_variance_}\n"
                f"  Explained variance ratio: {self.explained_variance_ratio_}\n"
                f"  Total variance explained: {self.explained_variance_ratio_.sum():.4f}"
            )


# Utility functions for PCA analysis

def plot_explained_variance(pca: PCA, figsize=(12, 5)):
    """
    Plot explained variance and cumulative explained variance.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    figsize : tuple, optional
        Figure size.
    """
    import matplotlib.pyplot as plt

    if pca.explained_variance_ratio_ is None:
        raise ValueError("PCA must be fitted first.")

    n_components = len(pca.explained_variance_ratio_)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot explained variance
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
            alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Explained Variance by Component', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(1, n_components + 1))
    ax1.grid(axis='y', alpha=0.3)

    # Plot cumulative explained variance
    ax2.plot(range(1, n_components + 1), cumulative_variance,
            marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, n_components + 1))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nVariance Summary:")
    print(f"  First PC explains: {pca.explained_variance_ratio_[0]:.2%}")
    if n_components > 1:
        print(f"  First 2 PCs explain: {cumulative_variance[1]:.2%}")
    print(f"  All {n_components} PCs explain: {cumulative_variance[-1]:.2%}")


def biplot(pca: PCA, X: np.ndarray, labels: Optional[np.ndarray] = None,
          feature_names: Optional[list] = None, figsize=(10, 8)):
    """
    Create a biplot showing both samples and features in PCA space.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    X : ndarray
        Original data.
    labels : ndarray, optional
        Labels for coloring points.
    feature_names : list, optional
        Names of features for arrows.
    figsize : tuple, optional
        Figure size.
    """
    import matplotlib.pyplot as plt

    if pca.n_components_ < 2:
        raise ValueError("Biplot requires at least 2 components.")

    # Transform data
    X_pca = pca.transform(X)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot samples
    if labels is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                           cmap='viridis', alpha=0.6, edgecolors='k', s=50)
        plt.colorbar(scatter, ax=ax, label='Labels')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6,
                  edgecolors='k', s=50)

    # Plot feature vectors
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(pca.n_features_)]

    # Scale factor for arrows
    scale = 3.0

    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0,
                scale * pca.components_[0, i],
                scale * pca.components_[1, i],
                head_width=0.1, head_length=0.1,
                fc='red', ec='red', alpha=0.7, linewidth=2)
        ax.text(scale * pca.components_[0, i] * 1.15,
               scale * pca.components_[1, i] * 1.15,
               feature, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=12)
    ax.set_title('PCA Biplot', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("PCA Implementation Test")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    X = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1]
    ])

    print(f"\nOriginal data shape: {X.shape}")
    print(f"Data:\n{X}")

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    print(f"\n{pca}")

    # Transform
    X_transformed = pca.transform(X)
    print(f"\nTransformed data shape: {X_transformed.shape}")
    print(f"Transformed data:\n{X_transformed}")

    # Inverse transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    print(f"\nReconstructed data:\n{X_reconstructed}")

    # Calculate reconstruction error
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}")

    # Test with variance threshold
    print("\n" + "=" * 60)
    print("Testing with variance threshold (0.95)")
    pca_95 = PCA(n_components=0.95)
    X_95 = pca_95.fit_transform(X)
    print(f"Components kept: {pca_95.n_components_}")
    print(f"Variance explained: {pca_95.explained_variance_ratio_.sum():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
