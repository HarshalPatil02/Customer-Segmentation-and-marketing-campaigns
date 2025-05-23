import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import plotly.express as px

st.title("Customer Segmentation & Marketing Campaign Analysis")
# Upload File
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid Excel file.")
        else:
            st.subheader("Uploaded Data Preview")
            st.dataframe(df)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")


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

# Train-Test Split and KMeans Evaluation
if st.checkbox("Show Train-Test Split Evaluation for K-Means"):
    X = df[features_to_scale]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)

    train_clusters = kmeans.predict(X_train)
    test_clusters = kmeans.predict(X_test)

    sil_train = silhouette_score(X_train, train_clusters)
    sil_test = silhouette_score(X_test, test_clusters)

    st.write(f"**Silhouette Score (Train):** {sil_train:.2f}")
    st.write(f"**Silhouette Score (Test):** {sil_test:.2f}")

    # Visualizing Clusters for Train
    st.write("### K-Means Clusters on Training Data")
    pca_train = PCA(n_components=2).fit_transform(X_train)
    pca_train_df = pd.DataFrame(pca_train, columns=["PC1", "PC2"])
    pca_train_df["Cluster"] = train_clusters
    fig = px.scatter(pca_train_df, x="PC1", y="PC2", color="Cluster", title="Train Set Clusters")
    st.plotly_chart(fig)

    # Visualizing Clusters for Test
    st.write("### K-Means Clusters on Test Data")
    pca_test = PCA(n_components=2).fit_transform(X_test)
    pca_test_df = pd.DataFrame(pca_test, columns=["PC1", "PC2"])
    pca_test_df["Cluster"] = test_clusters
    fig = px.scatter(pca_test_df, x="PC1", y="PC2", color="Cluster", title="Test Set Clusters")
    st.plotly_chart(fig)

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
    
    # Hierarchical Clustering (Agglomerative)
    agglo = AgglomerativeClustering(n_clusters=k)
    df_pca['Agglo_Cluster'] = agglo.fit_predict(df_pca[['PC1', 'PC2']])
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    df_pca['DBSCAN_Cluster'] = dbscan.fit_predict(df_pca[['PC1', 'PC2']])
    
    # Silhouette Scores
    kmeans_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Cluster'])
    agglo_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Agglo_Cluster'])
    
    if len(set(df_pca['DBSCAN_Cluster'])) > 1:
        dbscan_silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['DBSCAN_Cluster'])
    else:
        dbscan_silhouette = -1
    
    # Visualizing Clusters
    st.write("### K-Means Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)
    
    st.write(f"Silhouette Score: {kmeans_silhouette:.2f}")
    
    st.write("### Hierarchical Clustering (Agglomerative)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Agglo_Cluster'], palette='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("### DBSCAN Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['DBSCAN_Cluster'], palette='tab10', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Silhouette Score Comparison")
    silhouette_scores = pd.DataFrame({
        "Clustering Method": ["K-Means", "Hierarchical", "DBSCAN"],
        "Silhouette Score": [kmeans_silhouette, agglo_silhouette, dbscan_silhouette]
    })
    
    fig = px.bar(silhouette_scores, x="Clustering Method", y="Silhouette Score", 
                 title="Silhouette Score Comparison", color="Clustering Method", text="Silhouette Score")
    st.plotly_chart(fig)
    
    st.write(f"**K-Means Silhouette Score:** {kmeans_silhouette:.2f}")
    st.write(f"**Hierarchical Silhouette Score:** {agglo_silhouette:.2f}")
    st.write(f"**DBSCAN Silhouette Score:** {dbscan_silhouette:.2f} (Lower score due to potential noise points)")
    
    st.subheader("Income Distribution Across Customers")
    st.write("This bar chart shows the income distribution of customers, grouped into different income ranges. It helps identify the most common income levels in the dataset.")
    
    fig, ax = plt.subplots()
    sns.histplot(df["Income"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Customer Segmentation Count")
    st.write("This bar chart shows the number of customers in each segment after applying K-Means clustering.")
    
    if "cluster" not in df.columns:
        st.error("Error: 'cluster' column is missing. Please ensure clustering has been performed.")
    else:
        cluster_counts = df["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Customer Segmentation Count", color="Cluster")
        st.plotly_chart(fig)
    
    st.subheader("Customer Segmentation Proportion")
    st.write("This pie chart represents the proportion of customers in each cluster. It helps in understanding the distribution of different customer groups.")
    
    fig = px.pie(df, names="cluster", title="Customer Segments", 
                 hole=0.3,  
                 color_discrete_sequence=px.colors.qualitative.Set2,  
                 labels={"cluster": "Customer Segment"},  
                 template="plotly_white")  
    
    fig.update_traces(textinfo='percent+label', textfont_size=14) 
    st.plotly_chart(fig)
    
    st.subheader("Spending Distribution by Product Category")
    
    spending_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_spending = df[spending_columns].sum().reset_index()
    df_spending.columns = ["Product Category", "Total Spending"]
    
    fig = px.bar(df_spending, x="Product Category", y="Total Spending", title="Total Spending by Product Category", color="Product Category")
    st.plotly_chart(fig)

