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
    df = pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

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

    # Visualizing Clusters
    st.write("### K-Means Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)

    # Silhouette Score
    silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Cluster'])
    st.write(f"Silhouette Score: {silhouette:.2f}")

    # Income Distribution
    st.subheader("Income Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Income"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Cluster Visulization
    st.subheader("Clusters Visualization")
    
    st.write("Columns in DataFrame:", df.columns)
    st.write("First few rows:", df.head())
    df = df.dropna(subset=["MntWines", "MntMeatProducts", "cluster"])
    df["cluster"] = df["cluster"].astype(str)
    if "MntWines" in df.columns and "MntMeatProducts" in df.columns and "cluster" in df.columns:
    fig = px.scatter(df, x="MntWines", y="MntMeatProducts", color="cluster",
                     title="Customer Segmentation",
                     labels={"MntWines": "Amount Spent on Wine", "MntMeatProducts": "Amount Spent on Meat"},
                     size_max=10)
    st.plotly_chart(fig)
    else:
    st.error("Required columns missing in DataFrame!")

    fig = px.scatter(df, x="MntWines", y="MntMeatProducts", color="cluster")
    st.plotly_chart(fig)
    
    # Download Processed Data
    st.write("### Download Processed Data")
    df_pca.to_csv("clustered_data.csv", index=False)
    st.download_button("Download CSV", data=df_pca.to_csv(index=False), file_name="clustered_data.csv")







