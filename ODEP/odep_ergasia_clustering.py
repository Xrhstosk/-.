import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch

# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΤΟΙΜΑΣΙΑ (Όπως στο Μέρος Α)
file_path = r"C:\PLHROFORIKH\ODEP\Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Προεπεξεργασία αριθμητικών δεδομένων
if 'Blood Pressure' in df.columns:
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# Επιλογή αριθμητικών γνωρισμάτων (Αγνοούμε το Sleep Disorder)
features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
            'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
X = df[features]

# Scaling (Απαραίτητο για clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. K-MEANS & ELBOW METHOD ---
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (k-Means)')
#Προσθέτουν τον τίτλο στην κορυφή και τις ετικέτες στους δύο άξονες, ώστε όποιος βλέπει το γράφημα (π.χ. ο καθηγητής) να καταλαβαίνει αμέσως τι μετράμε.
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# --- 3. ΙΕΡΑΡΧΙΚΗ ΣΥΣΤΑΔΟΠΟΙΗΣΗ & ΔΕΝΔΡΟΓΡΑΜΜΑ ---

plt.figure(figsize=(12, 7))
#Ψάχνει να βρει ποια δύο σημεία μοιάζουν περισσότερο για να τα ενώσει σε ένα ζευγάρι.   (gia sch.linkage)
#Μετά ψάχνει το επόμενο ζευγάρι, και ούτω καθεξής, μέχρι όλα να καταλήξουν σε μια μεγάλη ομάδα.
linkage_matrix = sch.linkage(X_scaled, method='ward')
dendrogram = sch.dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Participants')
plt.ylabel('Euclidean Distances')
plt.show()

# --- 4. DBSCAN & K-DISTANCE GRAPH ---
# Για να βρούμε το eps, υπολογίζουμε την απόσταση από τους k-εγγύτερους γείτονες
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.title('k-distance Graph (DBSCAN)')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon (eps) value')
plt.grid()
plt.show()

# ΕΦΑΡΜΟΓΗ DBSCAN (Με μια ενδεικτική τιμή eps=1.2)
dbscan = DBSCAN(eps=1.2, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)
print(f"DBSCAN Clusters found: {np.unique(db_clusters)}")