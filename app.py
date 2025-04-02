import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Title
st.title("Marketing Campaign Clustering")

# Upload the dataset
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.subheader("Uploaded Data Preview")
    st.dataframe(df)  

    # Data Preprocessing
    df.drop(['ID', 'Year_Birth'], axis=1, inplace=True)
    df['Income'].fillna(df['Income'].median(), inplace=True)

    # Label Encoding
    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])
    df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

    # Normalization
    features_to_scale = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                         'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                         'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df[features_to_scale])
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

    # K-Means Clustering
    k = 4  
    kmeans = KMeans(n_clusters=k, random_state=0)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["MntWines", "MntMeatProducts"]])


    # Visualizing Clusters
    st.write("### K-Means Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)

    # Define features for clustering
    clustering_features = ["MntWines", "MntMeatProducts"]

    # ==============================
    # Hierarchical Clustering
    # ==============================
    st.subheader("Hierarchical Clustering")

    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
    df['hierarchical_cluster'] = hierarchical.fit_predict(df[clustering_features])

    fig, ax = plt.subplots()
    scatter = ax.scatter(df[clustering_features[0]], df[clustering_features[1]], c=df['hierarchical_cluster'], cmap='rainbow')
    plt.xlabel(clustering_features[0])
    plt.ylabel(clustering_features[1])
    plt.title('Hierarchical Clustering')
    st.pyplot(fig)

    # Silhouette Score
    silhouette_avg_hierarchical = silhouette_score(df[clustering_features], df['hierarchical_cluster'])
    st.write(f"Silhouette Score (Hierarchical Clustering): {silhouette_avg_hierarchical:.2f}")

    # ==============================
    # DBSCAN Clustering
    # ==============================
    st.subheader("DBSCAN Clustering")

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(df[clustering_features])

    # Remove noise points (-1) for Silhouette Score
    valid_labels = df['dbscan_cluster'][df['dbscan_cluster'] != -1]

    if len(set(valid_labels)) > 1:
        silhouette_avg_dbscan = silhouette_score(df[clustering_features][df['dbscan_cluster'] != -1], valid_labels)
        st.write(f"Silhouette Score (DBSCAN): {silhouette_avg_dbscan:.2f}")
    else:
        st.write("Silhouette Score (DBSCAN) cannot be computed due to too few clusters.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df[clustering_features[0]], df[clustering_features[1]], c=df['dbscan_cluster'], cmap='rainbow')
    plt.xlabel(clustering_features[0])
    plt.ylabel(clustering_features[1])
    plt.title('DBSCAN Clusters')
    st.pyplot(fig)


    # Income Distribution
    st.subheader("Income Distribution Across Customers")
    st.write("This bar chart shows the income distribution of customers, grouped into different income ranges. It helps identify the most common income levels in the dataset.")

    fig, ax = plt.subplots()
    sns.histplot(df["Income"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Bar Chart
    st.subheader("Customer Segmentation Count")
    st.write("This bar chart shows the number of customers in each segment after applying K-Means clustering.")

    if "cluster" not in df.columns:
        st.error("Error: 'cluster' column is missing. Please ensure clustering has been performed.")
    else:
        cluster_counts = df["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Customer Segmentation Count", color="Cluster")
        st.plotly_chart(fig)

    # Pie chart with data labels
    st.subheader("Customer Segmentation Proportion")
    st.write("This pie chart represents the proportion of customers in each cluster. It helps in understanding the distribution of different customer groups.")

    fig = px.pie(df, names="cluster", title="Customer Segments", 
                 hole=0.3,  
                 color_discrete_sequence=px.colors.qualitative.Set2,  
                 labels={"cluster": "Customer Segment"},  
                 template="plotly_white")  

    fig.update_traces(textinfo='percent+label', textfont_size=14) 
    st.plotly_chart(fig)





  







   

