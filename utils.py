import numpy as np

def normalize(x):
    return np.divide((x-x.mean(axis=0)), x.std(axis=0))

def PCA(x, d):
    N=x.shape[0]
    cov=x.T@x/N
    eigenvalues, eigenvectors=np.linalg.eigh(cov)
    sorted_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    A=eigenvectors[:,:d]
    y=x@A
    return y, eigenvalues, A

def MDS(delta, d):
    N=delta.shape[0]
    all_distances=np.sum(delta, axis=0)
    pair_sums=np.stack([all_distances for i in range(N)], axis=0)+np.stack([all_distances for i in range(N)], axis=1)
    G=-1/2*(delta+np.mean(delta)-1/N*pair_sums)
    eigenvalues, eigenvectors=np.linalg.eigh(G)
    sorted_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    y=np.multiply(eigenvectors[:,:d], np.sqrt(eigenvalues[:d]))
    return y, eigenvalues

def gaussian_kernel(x_i, x_j, width):
    return np.exp(-np.linalg.norm(x_i-x_j)**2/(2*width**2))