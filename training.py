import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump


# Melod data yang akan digunakan disini untuk sebagai pembatas menggunakan titik coma
# Load your dataset, specifying the delimiter and skipping bad lines
data = pd.read_csv('formresponden.csv', delimiter=';', error_bad_lines=False)

# data cleansing untuk select data apa saja yang menjadi variable untuk mengetahui fomo 
relevant_columns = ['jam_dalam_seminggu', 'X_Gadget', 'Y_Fomo']
data_clustering = data[relevant_columns].copy()

# Menghapus data yang gak digunakan drop 
data_clustering.dropna(subset=relevant_columns, inplace=True)

# Initialize StandardScaler untuk data training   
scaler = StandardScaler()

# Pastikan untuk menggunakan `data_clustering` bukan `data mentah `
scaled_data = scaler.fit_transform(data_clustering)

# Membuat training  data dengan model Kmeans model dengan number clustering 2 data yakni fomo dan tidak fomo 
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)

# Save the scaler
dump(scaler, 'scaler.joblib')

# Save the KMeans model
dump(kmeans, 'kmeans_model.joblib')
