    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    # Load Data
    df = pd.read_excel('marketing_campaign1.xlsx')

    # Data Preprocessing
    df['Income'].fillna(df['Income'].median(), inplace=True)
    df.drop(['ID', 'Year_Birth'], axis=1, inplace=True)

    # Normalize Selected Features
    features_to_scale = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
                     'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # PCA for Dimensionality Reduction
    pca = PCA(n_components=2)
    X_new = pd.DataFrame(pca.fit_transform(df[features_to_scale]), columns=['PC1', 'PC2'])

    # Sidebar for Cluster Selection
    st.sidebar.header("Clustering Parameters")
    k_pca = st.sidebar.slider("Select Number of Clusters (K-Means)", min_value=2, max_value=10, value=4)
    eps = st.sidebar.slider("DBSCAN: Epsilon (eps)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    min_samples = st.sidebar.slider("DBSCAN: Min Samples", min_value=2, max_value=10, value=5)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=k_pca, random_state=0)
    X_new['cluster_kmeans'] = kmeans.fit_predict(X_new)
    silhouette_kmeans = silhouette_score(X_new[['PC1', 'PC2']], X_new['cluster_kmeans'])

    # Hierarchical Clustering
    agglo = AgglomerativeClustering(n_clusters=k_pca, linkage='ward')
    X_new['cluster_hierarchical'] = agglo.fit_predict(X_new)
    silhouette_hierarchical = silhouette_score(X_new[['PC1', 'PC2']], X_new['cluster_hierarchical'])

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X_new['cluster_dbscan'] = dbscan.fit_predict(X_new)

    # Filter out noise points (-1)
    dbscan_filtered = X_new[X_new['cluster_dbscan'] != -1]
    if dbscan_filtered['cluster_dbscan'].nunique() > 1:
        silhouette_dbscan = silhouette_score(dbscan_filtered[['PC1', 'PC2']], dbscan_filtered['cluster_dbscan'])
    else:
        silhouette_dbscan = None

    # Display Silhouette Scores
    st.header("Silhouette Scores")
    st.metric(label="K-Means", value=f"{silhouette_kmeans:.2f}")
    st.metric(label="Hierarchical", value=f"{silhouette_hierarchical:.2f}")
    st.metric(label="DBSCAN", value=f"{silhouette_dbscan:.2f}" if silhouette_dbscan else "Not Computed")

    # Visualizations
    st.header("Clustering Visualizations")

    # K-Means Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_new['PC1'], y=X_new['PC2'], hue=X_new['cluster_kmeans'], palette='viridis', legend='full', alpha=0.7)
    plt.title("K-Means Clustering")
    st.pyplot(fig)

    # Hierarchical Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_new['PC1'], y=X_new['PC2'], hue=X_new['cluster_hierarchical'], palette='coolwarm', legend='full', alpha=0.7)
    plt.title("Hierarchical Clustering")
    st.pyplot(fig)

    # DBSCAN Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_new['PC1'], y=X_new['PC2'], hue=X_new['cluster_dbscan'], palette='rainbow', legend='full', alpha=0.7)
    plt.title("DBSCAN Clustering")
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







  







   

