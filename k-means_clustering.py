import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('your_file.csv')

grouped_data = data.groupby('userid').agg({'Price': 'sum'}).reset_index()

# filter the data with standard deviation method because k-means clustering approach is very sensitive to the outliers
mean = grouped_data['Price'].mean()
std_dev = grouped_data['Price'].std()
grouped_data['Price'] = grouped_data[(grouped_data['Price'] >= (mean - 3 * std_dev)) & (grouped_data['Price'] <= (mean + 3 * std_dev))]

X = grouped_data[['Price']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# define the cluster amount. here it is set as 10.
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_

grouped_data['Cluster'] = cluster_labels

cluster_stats = grouped_data.groupby('Cluster').agg({'userid': 'nunique', 'Price': ['min', 'max']})

for cluster in cluster_stats.index:
    distinct_user_count = cluster_stats.loc[cluster, ('userid', 'nunique')]
    min_price = cluster_stats.loc[cluster, ('Price', 'min')]
    max_price = cluster_stats.loc[cluster, ('Price', 'max')]
    print(f"Cluster {cluster} - Distinct User Count: {distinct_user_count}, Price Range: ({min_price}, {max_price})")