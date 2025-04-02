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

    # Silhouette Score
    silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Cluster'])
    st.write(f"Silhouette Score: {silhouette:.2f}")

    st.subheader("Hierarchical & DBSCAN Clustering")
    st.write("This section explores customer segmentation using K-Means, Hierarchical, and DBSCAN clustering methods.")

    # Prepare features for clustering
    features = ["MntWines", "MntMeatProducts"]
    df_cluster = df[features]

    # ==============================
    # Hierarchical Clustering
    # ==============================

    n_clusters_hierarchical = 3  # Define number of clusters
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
    df["cluster_hierarchical"] = hierarchical.fit_predict(df_cluster)

    # Compute Silhouette Score for Hierarchical Clustering
    silhouette_hierarchical = silhouette_score(df_cluster, df["cluster_hierarchical"])
    st.write(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical:.2f}")

    # Visualizing Hierarchical Clustering
    fig = px.scatter(df, x="MntWines", y="MntMeatProducts", color=df["cluster_hierarchical"].astype(str), 
                     title="Hierarchical Clustering", color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig)

    # ==============================
    # DBSCAN Clustering
    # ==============================

    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    df["cluster_dbscan"] = dbscan.fit_predict(df_cluster)

    # Exclude noise points (-1) from silhouette calculation
    valid_clusters = df[df["cluster_dbscan"] != -1]

    if len(valid_clusters["cluster_dbscan"].unique()) > 1:
        silhouette_dbscan = silhouette_score(valid_clusters[features], valid_clusters["cluster_dbscan"])
        st.write(f"Silhouette Score for DBSCAN Clustering: {silhouette_dbscan:.2f}")
    else:
        st.write("DBSCAN did not form enough clusters to compute a silhouette score.")

    # Visualizing DBSCAN Clustering
    fig = px.scatter(df, x="MntWines", y="MntMeatProducts", color=df["cluster_dbscan"].astype(str), 
                     title="DBSCAN Clustering (Noise = -1)", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)

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





  







   

