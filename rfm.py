import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

pipeline = joblib.load('model_pipeline.pkl')
categories_df = pd.read_csv('Categories.csv')
transactions_df = pd.read_csv('Transactions.csv')
df = pd.merge(transactions_df, categories_df, on='MerchantName', how='inner')

def calculate_rfm(df):
    rfm_df = df.groupby('CustomerID').agg({
        'CustomerLastTransactionFrom(days)': 'min',   # Recency 
        'TransactionRank': 'count',                  # Frequency
        'TransactionValue': 'sum'                    # Monetary value
    }).reset_index()
    
    # Renaming columns for clarity
    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Define a function to assign scores based on quantiles
    def rfm_score(x, quantiles):
        if x <= quantiles[0.25]:
            return 1
        elif x <= quantiles[0.50]:
            return 2
        elif x <= quantiles[0.75]:
            return 3
        else:
            return 4

    # Calculate quantiles for Recency, Frequency, and Monetary
    quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.5, 0.75]).to_dict()

    # Apply scoring for each metric
    rfm_df['R_Score'] = rfm_df['Recency'].apply(rfm_score, args=(quantiles['Recency'],))
    rfm_df['F_Score'] = rfm_df['Frequency'].apply(rfm_score, args=(quantiles['Frequency'],))
    rfm_df['M_Score'] = rfm_df['Monetary'].apply(rfm_score, args=(quantiles['Monetary'],))

    # Combine scores into a single RFM score
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    # Define segments based on RFM scores
    def segment_customer(df):
        if df['RFM_Score'] == '444':
            return 1  # Top Customers
        elif df['RFM_Score'] == '344':
            return 2  # Loyal Customers
        elif df['RFM_Score'] == '114':
            return 3  # At Risk
        else:
            return 4  # Potential Customers

    # Apply segmentation function
    rfm_df['Segment'] = rfm_df.apply(segment_customer, axis=1)

    # Adding the cluster to the RFM DataFrame
    rfm_df['Cluster'] = pipeline.predict(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Return the final RFM DataFrame
    return rfm_df

# Calculate the final RFM DataFrame
final_rfm_df = calculate_rfm(df)

# Define the get_top_merchants_by_category function
def get_top_merchants_by_category(customer_id, pipeline, df, final_df, n_categories=2, n_merchants=2):
    # Step 1: Filter for the specified customer from final_df which already contains RFM features
    customer_rfm = final_df[final_df['CustomerID'] == customer_id]
    
    if customer_rfm.empty:
        return f"No customer found with ID {customer_id}"
    
    # Extract only Recency, Frequency, and Monetary columns for prediction
    customer_rfm_features = customer_rfm[['Recency', 'Frequency', 'Monetary']]
    
    # Step 2: Predict the cluster for this customer using their RFM data
    customer_cluster = customer_rfm['Cluster'].values[0]
    
    # Step 3: Filter the dataset for customers in the same cluster
    same_cluster_customers = final_rfm_df[final_rfm_df['Cluster'] == customer_cluster]
    
    # Step 4: Aggregate transaction values by Category first to get the top categories
    top_categories = (df[df['CustomerID'].isin(same_cluster_customers['CustomerID'])]
                      .groupby('Category')['TransactionValue']
                      .mean()  # Using mean to get the average spending per customer in the same cluster
                      .reset_index()
                      .sort_values(by='TransactionValue', ascending=False)
                      .head(n_categories))  # Get the top `n_categories` (e.g., 2 categories)
    
    # Step 5: For each top category, find the top merchants
    top_merchants = {}
    
    for category in top_categories['Category']:
        top_merchant = (df[(df['Category'] == category) & (df['CustomerID'].isin(same_cluster_customers['CustomerID']))]
                        .groupby('MerchantName')['TransactionValue']
                        .mean()  # Using mean for average transaction per merchant in this category
                        .reset_index()
                        .sort_values(by='TransactionValue', ascending=False)
                        .head(n_merchants))  # Get the top `n_merchants` per category
        
        if not top_merchant.empty:
            top_merchants[category] = top_merchant['MerchantName'].tolist()  # Store merchants in a dictionary
    
    # Step 6: Return the top merchants with their respective categories
    return top_merchants

# Streamlit app
st.title("Customer Merchant Recommendation App")

customer_id = st.number_input("Enter Customer ID:", min_value=0)

if st.button("Get Recommendations"):
    if customer_id:
        top_merchants_by_category = get_top_merchants_by_category(customer_id, pipeline, df, final_rfm_df, n_categories=2, n_merchants=2)
        
        if isinstance(top_merchants_by_category, dict) and top_merchants_by_category:
            st.write("Top Merchants by Category:")
            for category, merchants in top_merchants_by_category.items():
                st.subheader(f"Category: {category}")
                st.write(", ".join(merchants))
        else:
            st.write(top_merchants_by_category)  # Display message if customer not found
    else:
        st.write("Please enter a valid Customer ID.")
