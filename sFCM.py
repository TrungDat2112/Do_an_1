import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
import tkinter as tk
from scipy.special import comb

np.random.seed(0)
file_path = 'iris.data'
column_name = ['sepal_length', 'sepal_width', 'petal_length', 'prtal_width', 'species']
data = pd.read_csv(file_path, header=None, names = column_name)
species_to_cluster = {species: idx for idx, species in enumerate(data['species'].unique())}
data['cluster'] = data['species'].map(species_to_cluster)
true_labels = data['cluster'].values
n_samples = data.shape[0]
n_feature = data.shape[1] - 2
n_clusters = len(data['species'].unique())

def init_U_hat(c):
    U_hat = np.zeros((n_samples,n_clusters))
    x = np.random.choice(range(0,n_samples),size = 15, replace=False)
    for index, row in data.iterrows():
        U_hat[index, row['cluster']] = 1.0

    zero_matrix = np.zeros((n_samples,(c-n_clusters)))
    U_middle = U_hat * 0.51

    U_final = np.hstack((U_middle,zero_matrix))
    K = np.zeros((n_samples,c))
    for j in x:
        K[j] = U_final[j]
    return K

def initialize_centers(X, c):
    indices = np.random.choice(n_samples, c, replace=False)
    return X[indices]

def update_U(X, V, U_hat, m, c):
    U = np.zeros((n_samples,c))
    for k in range(n_samples):
        d_ki = np.linalg.norm(X[k] - V, axis=1)**2
        if m > 1:
            denom = np.sum([(1 / d) ** (1 / (m - 1)) if d > 0 else np.inf for d in d_ki])
            for i in range(c):
                if d_ki[i] == 0:
                    U[k,i] = 1
                else:
                    U[k, i] = U_hat[k, i] + ((1 - np.sum(U_hat[k])) * ((1 / d_ki[i]) ** (1 / (m - 1))) / denom)
        else:
            min_index = np.argmin(d_ki)
            U[k, min_index] = U_hat[k, min_index] + 1 - np.sum(U_hat[k])
    
    return U

def update_centers(U, U_hat, X, m, c):
    V = np.zeros((c, n_feature))
    
    for i in range(c):
        num = np.sum([((U[k, i] - U_hat[k, i]) ** m) * X[k] for k in range(n_samples)], axis=0)
        denom = np.sum([(U[k, i] - U_hat[k, i]) ** m for k in range(n_samples)])
        V[i] = num / denom
    
    return V

def sFCM(X, U_hat, c, m, epsilon, max_iter=100 ):
    V = initialize_centers(X, c)
    U = np.zeros((n_samples,c))
    for _ in range(max_iter):
        U_old = np.copy(U)
        
        U = update_U(X, V, U_hat, m, c)
        V = update_centers(U, U_hat, X, m, c)
        
        if np.linalg.norm(U - U_old) < epsilon:
            break
    
    return U, V

def calculate_jm(X, U, centers, m):
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
        Hc = np.nan_to_num(Hc) 
    return Hc

def dunn_index(X, labels):
    distances = squareform(pdist(X, 'euclidean'))
    n_clusters = len(np.unique(labels))
    intra_cluster_distances = np.zeros(n_clusters)
    inter_cluster_distances = np.full((n_clusters, n_clusters), np.inf)
    for i in range(n_clusters):
        cluster_i = X[labels == i]
        if cluster_i.size:
            intra_cluster_distances[i] = np.max(pdist(cluster_i, 'euclidean'))

    # Calculate inter-cluster distances
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_distances = distances[np.ix_(labels == i, labels == j)]
            if inter_distances.size:
                inter_cluster_distances[i, j] = np.min(inter_distances)

    min_inter = np.min(inter_cluster_distances[inter_cluster_distances != np.inf])
    max_intra = np.max(intra_cluster_distances)
    dunn_index = min_inter / max_intra if max_intra > 0 else 0

    return dunn_index

def alternative_silhouette_samples(X, labels, epsilons=1e-6):
    distances = pairwise_distances(X)
    unique_labels = np.unique(labels)
    silhouette_samples = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        same_cluster = (labels == labels[i])
        a_pj = np.mean(distances[i, same_cluster])
        min_distance = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = (labels == label)
                distance_to_other_cluster = np.mean(distances[i, other_cluster])
                if distance_to_other_cluster < min_distance:
                    min_distance = distance_to_other_cluster
        b_pj = min_distance
        silhouette_samples[i] = (b_pj - a_pj) / (a_pj + epsilons)

    return silhouette_samples

def calculate_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def alternative_simplified_silhouette_samples(X, labels, centroids, epsilon=1e-5):
    distances_to_centroids = pairwise_distances(X, centroids)
    intra_distances = distances_to_centroids[np.arange(len(labels)), labels]
    inter_distances = np.where(
        (np.arange(len(centroids)) == labels[:, None]),
        np.inf,
        distances_to_centroids
    )
    nearest_other_distance = np.min(inter_distances, axis=1)

    silhouette_values = (nearest_other_distance - intra_distances) / (intra_distances + epsilon)

    return silhouette_values

def contingency_table(labels_true, labels_pred):
    n_samples = len(labels_true)
    contingency = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        for j in range(n_samples):
            contingency[i, j] = (labels_true[i] == labels_true[j]) and (labels_pred[i] == labels_pred[j])
    return contingency

# Tính Rand Index
def rand_index(labels_true, labels_pred):
    contingency = contingency_table(labels_true, labels_pred)
    tp_plus_fp = comb(np.sum(contingency, axis=1), 2).sum()
    tp_plus_fn = comb(np.sum(contingency, axis=0), 2).sum()
    tp = comb(contingency, 2).sum()
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(labels_true), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

# Tính Adjusted Rand Index
def adjusted_rand_index(labels_true, labels_pred):
    contingency = contingency_table(labels_true, labels_pred)
    sum_comb_c = comb(np.sum(contingency, axis=1), 2).sum()
    sum_comb_k = comb(np.sum(contingency, axis=0), 2).sum()
    sum_comb = comb(contingency, 2).sum()
    n = len(labels_true)
    index = sum_comb
    expected_index = sum_comb_c * sum_comb_k / comb(n, 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    return (index - expected_index) / (max_index - expected_index)

# Tính Jaccard Coefficient
def jaccard_coefficient(labels_true, labels_pred):
    contingency = contingency_table(labels_true, labels_pred)
    tp_plus_fp = comb(np.sum(contingency, axis=1), 2).sum()
    tp_plus_fn = comb(np.sum(contingency, axis=0), 2).sum()
    tp = comb(contingency, 2).sum()
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    return tp / (tp + fp + fn)

root = tk.Tk()
root.title("SFCM")
root.geometry("400x400")
frame_1 = tk.Frame(root, width=200, height=200)
frame_1.grid(row=0, column=0)
tk.Label(frame_1, text="Number of Clusters:").grid(row=0, column=0, sticky='w')
num_clusters = tk.IntVar()
num_clusters.set(2)  # default value
tk.Entry(frame_1, textvariable=num_clusters).grid(row=0, column=1)

# Exponent for the clustering
tk.Label(frame_1, text="Exponent for the clustering (usually 2):").grid(row=1, column=0, sticky='w')
exponent = tk.DoubleVar()
exponent.set(2.0)  # default value
tk.Entry(frame_1, textvariable=exponent).grid(row=1, column=1)

tk.Label(frame_1, text = "epsilone").grid(row = 2, column=0, sticky='w')
epsilone = tk.DoubleVar()
epsilone.set(0.01)
tk.Entry(frame_1, textvariable=epsilone).grid(row=2,column=1)

def run_sFCM():
    X = data.iloc[:, :4].values
    epsilon = epsilone.get()
    m = exponent.get()
    c = num_clusters.get()
    U_hat = init_U_hat(c)
    U, V = sFCM(X, U_hat, c, m, epsilon,max_iter=100)
    cluster_labels = np.argmax(U, axis=1)
    Jm = calculate_jm(X, U, V, m)
    Fc = calculate_fc(U)
    Hc = calculate_hc(U)
    One_minus_Fc = 1 - Fc

    vrc_score = calinski_harabasz_score(X, cluster_labels)

    db_score = davies_bouldin_score(X, cluster_labels)
    dn_index = dunn_index(X, cluster_labels)

    silhouette_vals = silhouette_samples(X, cluster_labels)

    avg_silhouette = silhouette_score(X, cluster_labels)
    epsilons = 1e-6
    alt_silhouette_vals = alternative_silhouette_samples(X, cluster_labels, epsilons)
    avg_alt_silhouette = np.mean(alt_silhouette_vals)
    centroids = calculate_centroids(X, cluster_labels, k=np.unique(cluster_labels).size)
    asswc_values = alternative_simplified_silhouette_samples(X, cluster_labels, centroids)
    mean_asswc = np.mean(asswc_values)
    Rand_index =  rand_index(true_labels, cluster_labels)
    Adjusted_Rand_Index = adjusted_rand_index(true_labels, cluster_labels)
    Jaccard_Coefficient = jaccard_coefficient(true_labels, cluster_labels)
    frame_2 = tk.Frame(root, width=200, height=200)
    frame_2.grid(row=0, column=1, sticky='nsew')
    label_cl = tk.Label(frame_2, text="Cluster centers", font=("Courier", 14))
    label_cl.pack() 
    text_cl = tk.Text(frame_2, height=12, width=52)
    text_cl.pack(expand=True, fill='both') 
    full_textcl = f"Cluster centers:\n{V}"
    text_cl.insert(tk.END, full_textcl)
    frame_3 = tk.Frame(root, width=200, height=200)
    frame_3.grid(row=1, column=0, sticky='nsew')
    l = tk.Label(frame_3, text="Result", font=("Courier", 14))
    l.pack() 
    T = tk.Text(frame_3, height=12, width=52)
    T.pack(expand=True, fill='both') 
    full_text = f"Jm: {Jm}\nFc: {Fc}\nHc: {Hc}\n1 - Fc: {One_minus_Fc}\nVRC: {vrc_score}\nDB: {db_score}\nDunn: {dn_index}\nSWC: {avg_silhouette}\nASWC: {avg_alt_silhouette}\nw: {Rand_index}\nwA: {Adjusted_Rand_Index}\nwJ: {Jaccard_Coefficient}"
    T.insert(tk.END, full_text)
    frame_4 = tk.Frame(root, width=200, height=200)
    frame_4.grid(row=1, column=1)
    listbox = tk.Listbox(frame_4, height=10, width=50)
    scrollbar = tk.Scrollbar(frame_4)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    for i in range(X.shape[0]):
        listbox.insert(tk.END, f"Item {i+1} : {cluster_labels[i]}")
    
run_button = tk.Button(frame_1, text="Run Clustering", command=run_sFCM)
run_button.grid(row=3, columnspan=2, pady=10)
root.mainloop()
