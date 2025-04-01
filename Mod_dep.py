{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2f111891-7af1-4bf2-a863-aa98d8e8bd98",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-01 21:46:44.625 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Features of the Streamlit App\n",
    "# ✅ Upload customer data (CSV)\n",
    "# ✅ Preprocess and segment customers using K-Means\n",
    "# ✅ Display cluster insights & visualizations\n",
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Load saved models\n",
    "kmeans = joblib.load(\"kmeans_model.pkl\")\n",
    "pca = joblib.load(\"pca_model.pkl\")\n",
    "\n",
    "# Streamlit UI\n",
    "def main():\n",
    "    st.title(\"Customer Segmentation App\")\n",
    "    st.write(\"Upload customer data to get segmentation insights\")\n",
    "    \n",
    "    # File uploader\n",
    "    uploaded_file = st.file_uploader(\"Upload CSV File\", type=[\"csv\"])\n",
    "    \n",
    "    if uploaded_file is not None:\n",
    "        df = pd.read_csv(uploaded_file)\n",
    "        \n",
    "        # Select relevant features\n",
    "        features = [\"Income\", \"MntWines\", \"MntFruits\", \"MntMeatProducts\", \n",
    "                    \"MntFishProducts\", \"MntSweetProducts\", \"MntGoldProds\", \n",
    "                    \"NumWebPurchases\", \"NumCatalogPurchases\", \"NumStorePurchases\", \n",
    "                    \"NumWebVisitsMonth\"]\n",
    "        df = df[features]\n",
    "        \n",
    "        # Scale and apply PCA\n",
    "        df_scaled = scaler.transform(df)\n",
    "        df_pca = pca.transform(df_scaled)\n",
    "        \n",
    "        # Predict clusters\n",
    "        clusters = kmeans.predict(df_pca)\n",
    "        df[\"Cluster\"] = clusters\n",
    "        \n",
    "        st.write(\"### Clustered Data Sample\")\n",
    "        st.write(df.head())\n",
    "        \n",
    "        # Visualization\n",
    "        st.write(\"### Cluster Distribution\")\n",
    "        fig, ax = plt.subplots()\n",
    "        sns.countplot(x=df[\"Cluster\"], palette=\"coolwarm\", ax=ax)\n",
    "        st.pyplot(fig)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n",
    "\n",
    "# Next Steps\n",
    "# 1. Run the Streamlit App:\n",
    "\n",
    "# bash\n",
    "# Copy code\n",
    "# streamlit run streamlit_customer_segmentation.py\n",
    "# 2. Upload a CSV file containing customer data.\n",
    "# 3. View segmentation insights & visualizations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90ba93a1-028e-419d-847f-27fad8d0055a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
