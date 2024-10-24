import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (Mall Customers dataset)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
data = pd.read_csv(url)

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set1')
plt.title("Customer Segmentation using K-Means")
plt.show()
