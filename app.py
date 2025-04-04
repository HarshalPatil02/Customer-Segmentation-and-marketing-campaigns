import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px

st.title("Customer Segmentation & Marketing Campaign Analysis")

# Upload File
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.subheader("Uploaded Data Preview")
    st.dataframe(df)

    # Data Preprocessing
    if 'ID' in df.columns and 'Year_Birth' in df.columns:
        df.drop(['ID', 'Year_Birth'], axis=1, inplace=True)

    if 'Income' in df.columns:
        df['Income'].fillna(df['Income'].median(), inplace=True)

    # Label Encoding
    le = LabelEncoder()
    if 'Education' in df.columns and 'Marital_Status' in df.columns:
        df['Education'] = le.fit_transform(df['Education'])
        df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

    # Normalization
    features_to_scale = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                         'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                         'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    
    missing_cols = [col for col in features_to_scale if col not in df.columns]
    if not missing_cols:
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df[features_to_scale])
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca)

    # Hierarchical Clustering
    agglo = AgglomerativeClustering(n_clusters=4)
    df_pca['Agglo_Cluster'] = agglo.fit_predict(df_pca[['PC1', 'PC2']])

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    df_pca['DBSCAN_Cluster'] = dbscan.fit_predict(df_pca[['PC1', 'PC2']])

    # Silhouette Scores
    kmeans_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Cluster'])
    agglo_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Agglo_Cluster'])
    dbscan_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['DBSCAN_Cluster']) if len(set(df_pca['DBSCAN_Cluster'])) > 1 else -1

    # Plot Results
    st.write(f"**K-Means Silhouette Score:** {kmeans_silhouette:.2f}")
    st.write(f"**Hierarchical Silhouette Score:** {agglo_silhouette:.2f}")
    st.write(f"**DBSCAN Silhouette Score:** {dbscan_silhouette:.2f}")

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

    # Spending Distribution by Product category
    st.subheader("Spending Distribution by Product Category")

    spending_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_spending = df[spending_columns].sum().reset_index()
    df_spending.columns = ["Product Category", "Total Spending"]

    fig = px.bar(df_spending, x="Product Category", y="Total Spending", title="Total Spending by Product Category", color="Product Category")
    st.plotly_chart(fig)

    # Recency vs. Spending Behavior
    spending_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spending"] = df[spending_columns].sum(axis=1)

    st.subheader("Recency vs. Spending Behavior")

    fig = px.scatter(df, x="Recency", y="Total_Spending", title="Recency vs. Total Spending", color="cluster", size="Total_Spending", hover_data=['Income'])
    st.plotly_chart(fig)

   

