import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class SpectralClustering:
    def __init__(self, k, data):
        self.k = k
        self.data = data


    def fit_predict(self, L_aliq):
        eigenvalues, eigenvectors = np.linalg.eig(L_aliq)
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        H_matrix = sorted_eigenvectors[:, :self.k]
        H_matrix = np.real(H_matrix)
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(H_matrix)
        labels = kmeans.labels_
        return labels


    def similarity_matrix(self, sigma2 = 0.1):
        similarity_matrix = np.zeros((self.data.shape[0], self.data.shape[0]))

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                similarity_matrix[i,j] = np.exp(-(np.linalg.norm(self.data[i]-self.data[j]))**2 / sigma2)

        return similarity_matrix



    def laplacian_matrix(self, similarity_matrix):
        degree_matrix = np.diag(np.sum(similarity_matrix, axis = 1))
        L_aliq = np.eye(self.data.shape[0]) - np.linalg.inv(degree_matrix)@similarity_matrix

        return L_aliq




    

