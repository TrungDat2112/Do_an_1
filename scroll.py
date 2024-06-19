import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from scipy.spatial.distance import cdist
import tkinter as tk

# Load the CSV file to check the first few rows and its structure
file_path = 'iris.data'
data = pd.read_csv(file_path, header=None)  # Assuming no header in the file

def initialize_membership_matrix(n_samples, n_clusters):
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    return U

def calculate_cluster_centers(X, U, m):
    um = U ** m
    centers = um.T @ X / um.sum(axis=0)[:, None]
    return centers

def update_membership_matrix(X, centers, m, A=None):
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    p = 2 / (m - 1)
    distances = np.zeros((n_samples, n_clusters))

    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(X - centers[i], axis=1, ord=2)
    U = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        for j in range(n_samples):
            denominator = np.sum((distances[j, i] / distances[j, :]) ** p)
            U[j, i] = 1 / denominator

    return U
def calculate_jm(X, U, centers, m):
    n_samples, n_features = X.shape
    n_clusters = centers.shape[0]
    Jm = 0
    for i in range(n_clusters):
        diff = X - centers[i]
        dist_squared = np.sum(diff ** 2, axis=1)
        Jm += np.sum((U[:, i] ** m) * dist_squared)
    return Jm
def calculate_fc(U):
    Fc = np.sum(U ** 2) / U.shape[0]
    return Fc

def calculate_hc(U):
    with np.errstate(divide='ignore', invalid='ignore'):
        Hc = -np.sum(U * np.log(U)) / U.shape[0]
        Hc = np.nan_to_num(Hc)  # Replace nan with 0 and inf with large finite numbers
    return Hc

def fcm(X, n_clusters, m, epsilon, max_iter=100, A=None):
    n_samples = X.shape[0]
    U = initialize_membership_matrix(n_samples, n_clusters)
    for iteration in range(max_iter):
        U_old = U.copy()
        centers = calculate_cluster_centers(X, U, m)
        U = update_membership_matrix(X, centers, m)
        if np.linalg.norm(U - U_old) < epsilon:
            break

    return U, centers

def dunn_index(X, labels):
    # Calculate pairwise distances between points
    distances = squareform(pdist(X, 'euclidean'))
    n_clusters = len(np.unique(labels))

    # Initialize diameters to zero
    intra_cluster_distances = np.zeros(n_clusters)

    # Initialize inter-cluster distances to a large number
    inter_cluster_distances = np.full((n_clusters, n_clusters), np.inf)

    # Calculate intra-cluster distances (diameters)
    for i in range(n_clusters):
        cluster_i = X[labels == i]
        if cluster_i.size:
            intra_cluster_distances[i] = np.max(pdist(cluster_i, 'euclidean'))

    # Calculate inter-cluster distances
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Filter distances between different clusters
            inter_distances = distances[np.ix_(labels == i, labels == j)]
            if inter_distances.size:
                inter_cluster_distances[i, j] = np.min(inter_distances)

    # Calculate Dunn's index
    min_inter = np.min(inter_cluster_distances[inter_cluster_distances != np.inf])
    max_intra = np.max(intra_cluster_distances)
    dunn_index = min_inter / max_intra if max_intra > 0 else 0

    return dunn_index

def alternative_silhouette_samples(X, labels, epsilons=1e-6):
    distances = pairwise_distances(X)
    unique_labels = np.unique(labels)
    silhouette_samples = np.zeros(X.shape[0])

    # Compute a_pj and b_pj for each sample
    for i in range(X.shape[0]):
        # Distances from the i-th sample to all other points in the same cluster
        same_cluster = (labels == labels[i])
        a_pj = np.mean(distances[i, same_cluster])

        # Find the smallest mean distance to all points in any other cluster
        min_distance = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = (labels == label)
                distance_to_other_cluster = np.mean(distances[i, other_cluster])
                if distance_to_other_cluster < min_distance:
                    min_distance = distance_to_other_cluster
        b_pj = min_distance

        # Calculate the alternative silhouette value for sample i
        silhouette_samples[i] = (b_pj - a_pj) / (a_pj + epsilons)

    return silhouette_samples

def calculate_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def alternative_simplified_silhouette_samples(X, labels, centroids, epsilon=1e-5):
    distances_to_centroids = pairwise_distances(X, centroids)
    intra_distances = distances_to_centroids[np.arange(len(labels)), labels]

    # Avoid the use of intra-cluster distances in finding the nearest other cluster
    inter_distances = np.where(
        (np.arange(len(centroids)) == labels[:, None]),
        np.inf,
        distances_to_centroids
    )
    nearest_other_distance = np.min(inter_distances, axis=1)

    silhouette_values = (nearest_other_distance - intra_distances) / (intra_distances + epsilon)

    return silhouette_values



root = tk.Tk()
root.title("FCM")
root.geometry("400x400")
frame_1 = tk.Frame(root, width=200, height=200)
frame_1.grid(row=0, column=0)
tk.Label(frame_1, text="Number of Clusters:").grid(row=0, column=0, sticky='w')
num_clusters = tk.IntVar()
num_clusters.set(2)  # default value
tk.Entry(frame_1, textvariable=num_clusters).grid(row=0, column=1)

tk.Label(frame_1, text="Exponent for the clustering (usually 2):").grid(row=1, column=0, sticky='w')
exponent = tk.DoubleVar()
exponent.set(2.0)  # default value
tk.Entry(frame_1, textvariable=exponent).grid(row=1, column=1)

tk.Label(frame_1, text = "epsilone").grid(row = 2, column=0, sticky='w')
epsilone = tk.DoubleVar()
epsilone.set(0.01)
tk.Entry(frame_1, textvariable=epsilone).grid(row=2,column=1)
def run_clustering():
    X = data.iloc[:, :4].values
    epsilon = epsilone.get()
    m = exponent.get()
    n_clusters = num_clusters.get()
    U, centers = fcm(X, n_clusters, m, epsilon, max_iter=100, A=None)
    cluster_labels = np.argmax(U, axis=1)
    Jm = calculate_jm(X, U, centers, m)
    Fc = calculate_fc(U)
    Hc = calculate_hc(U)
    One_minus_Fc = 1 - Fc

    vrc_score = calinski_harabasz_score(X, cluster_labels)

    db_score = davies_bouldin_score(X, cluster_labels)



    # Usage
    dn_index = dunn_index(X, cluster_labels)

    silhouette_vals = silhouette_samples(X, cluster_labels)

    avg_silhouette = silhouette_score(X, cluster_labels)



    # Compute the alternative silhouette samples
    epsilons = 1e-6
    alt_silhouette_vals = alternative_silhouette_samples(X, cluster_labels, epsilons)
    avg_alt_silhouette = np.mean(alt_silhouette_vals)


    centroids = calculate_centroids(X, cluster_labels, k=np.unique(cluster_labels).size)


    # Computing the ASSWC for each sample
    asswc_values = alternative_simplified_silhouette_samples(X, cluster_labels, centroids)

    # Computing the mean ASSWC across all samples for the overall score
    mean_asswc = np.mean(asswc_values)

    # Tạo frame ở góc trên bên phải
    frame_2 = tk.Frame(root, width=200, height=200)
    frame_2.grid(row=0, column=1, sticky='nsew')
    label_cl = tk.Label(frame_2, text="Cluster centers", font=("Courier", 14))
    label_cl.pack()  # Pack the label into frame_2
    text_cl = tk.Text(frame_2, height=12, width=52)
    text_cl.pack(expand=True, fill='both')  # Pack the text widget into frame_2
    full_textcl = f"Cluster centers:\n{centers}"
    text_cl.insert(tk.END, full_textcl)

    # Update frame_3 with the clustering results
    frame_3 = tk.Frame(root, width=200, height=200)
    frame_3.grid(row=1, column=0, sticky='nsew')
    l = tk.Label(frame_3, text="Result", font=("Courier", 14))
    l.pack()  # Pack the label into frame_3
    T = tk.Text(frame_3, height=12, width=52)
    T.pack(expand=True, fill='both')  # Pack the text widget into frame_3
    full_text = f"Jm: {Jm}\nFc: {Fc}\nHc: {Hc}\n1 - Fc: {One_minus_Fc}\nVRC: {vrc_score}\nDB: {db_score}\nDunn: {dn_index}\nSWC: {avg_silhouette}\nASWC: {avg_alt_silhouette}\nASSWC: {mean_asswc}"
    T.insert(tk.END, full_text)


    # Tạo frame ở góc dưới bên phải
    frame_4 = tk.Frame(root, width=200, height=200)
    frame_4.grid(row=1, column=1)
    listbox = tk.Listbox(frame_4, height=10, width=50)
    scrollbar = tk.Scrollbar(frame_4)

    # Đóng gói Scrollbar vào Frame
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Đóng gói Listbox vào Frame
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)

    # Liên kết Scrollbar với Listbox
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    for i in range(X.shape[0]):
        listbox.insert(tk.END, f"Item {i+1} : {cluster_labels[i]}")
    
run_button = tk.Button(frame_1, text="Run Clustering", command=run_clustering)
run_button.grid(row=3, columnspan=2, pady=10)
root.mainloop()
