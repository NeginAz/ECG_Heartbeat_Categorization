import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReductionVisualization:
    def __init__(self, label_column='label', random_state=42):
        self.label_column = label_column
        self.random_state = random_state

    def plot_pca(self, data: pd.DataFrame, target_class=None, n_components=2):
        """Applies PCA and visualizes the data for a specific target class or all classes."""
        plt.figure(figsize=(8, 2))
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        
        if target_class is not None:
            class_data = data[data[self.label_column] == target_class]
            X_pca = pca.fit_transform(class_data.drop(columns=[self.label_column]))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, label=f'Class {target_class}')
            plt.title(f'PCA Visualization of Class {target_class} with {n_components} Components')
        else:
            X_pca = pca.fit_transform(data.drop(columns=[self.label_column]))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=data[self.label_column], cmap='tab10')
            plt.title(f'PCA Visualization of All Classes')
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_tsne(self, data: pd.DataFrame, target_class=None, n_components=2):
        """Applies t-SNE and visualizes the data for a specific target class or all classes."""
        plt.figure(figsize=(8, 2))
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, random_state=self.random_state)
        
        if target_class is not None:
            class_data = data[data[self.label_column] == target_class]
            X_tsne = tsne.fit_transform(class_data.drop(columns=[self.label_column]))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, label=f'Class {target_class}')
            plt.title(f't-SNE Visualization of Class {target_class} with {n_components} Components')
        else:
            X_tsne = tsne.fit_transform(data.drop(columns=[self.label_column]))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, c=data[self.label_column], cmap='tab10')
            plt.title(f't-SNE Visualization of All Classes')
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_classes(self, data: pd.DataFrame, method='pca', n_components_map=None, default_components=2):
        """
        Plots each class individually using PCA or t-SNE with customizable component numbers.
        
        Parameters:
        - data (pd.DataFrame): The input data.
        - method (str): 'pca' or 'tsne' for the visualization method.
        - n_components_map (dict): A mapping of {class_label: n_components}.
        - default_components (int): Default number of components if not specified in the map.
        """
        unique_classes = data[self.label_column].unique()
        
        for target_class in unique_classes:
            n_components = n_components_map.get(target_class, default_components) if n_components_map else default_components
            if method.lower() == 'pca':
                self.plot_pca(data, target_class=target_class, n_components=n_components)
            elif method.lower() == 'tsne':
                self.plot_tsne(data, target_class=target_class, n_components=n_components)
            else:
                print("Invalid method. Choose 'pca' or 'tsne'.")
