# Problem Description 
The primary objective of this analysis is to identify and understand target customer segments that can enable the marketing team to develop data-driven strategies and targeted campaigns. By clustering customers into distinct groups, this analysis aims to provide actionable insights that can optimize marketing resource allocation and improve customer engagement.

Unsupervised learning (clustering) is appropriate because the dataset does not contain predefined labels or customer segment categories. Clustering techniques allow natural groupings to emerge from the data based on similarities in age, income, and spending behavior, making them well-suited for exploratory market segmentation.

# Dataset Description
The dataset used is the *Mall Customer Segmentation Dataset* from Kaggle, consisting of 200 mall customers.

- Features:
   - `CustomerID`: Unique identifier for each customer  
   - `Age`: Customer age in years  
   - `Gender`: Customer gender  
   - `Annual Income`: Customer annual income
   - `Spending Score`: A mall-assigned score representing customer spending behavior

## Data Preprocessing:
Column names for Annual Income and Spending Score were standardized to remove special characters for easier programmatic access. Annual income values were converted from thousands of dollars (k$) to dollar amounts for improved interpretability. Data quality checks revealed no missing values or significant outliers in the age variable, so no imputation or outlier treatment was required.


# Setup Instructions
## How to install dependencies
Install the required Python libraries using the requirements.txt file:

``` pip install -r requirements.txt ```

## How to get the data
Download the dataset from this Kaggle Link: 
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### How to run the notebook
Launch Jupyter Notebook or JupyterLab and run the provided ```analysis.ipynb``` notebook from top to bottom to reproduce all results, visualizations, and findings.

# Results Summary
## Optimal K chosen
The optimal number of clusters was determined to be K = 6 using a combination of the elbow method and silhouette score analysis. At K=6, the silhouette score reached 0.431, indicating reasonable cluster cohesion, while the inertia curve began to plateau (inertia = 134.353), suggesting diminishing returns from additional clusters.

## Cluster Descriptions
The six clusters represent diverse customer segments with varying combinations of age, income, and spending behavior. The clusters are not sharply separated, instead exhibiting gradual transitions, which suggests overlapping customer characteristics rather than rigid segment boundaries.

## Number and types of anomalies found
An Isolation Forest model with a contamination value of 0.05 identified 10 anomalies (5% of the dataset).
- No anomalies were found in Clusters 0 and 3
- Clusters 1, 2, and 4 contained 2 anomalies each
- Cluster 5 contained 4 anomalies

These anomalies were categorized into three profiles:
- VIPs: High-income, high-spending customers (Clusters 1 and 4)
- Young Shoppers: Young, high-spending customers with lower income (Cluster 2)
- Disengaged Customers: Mostly older customers with very low spending scores (Cluster 5)

## Key business insights
The integrated clustering and dimensionality reduction analysis reveals that customer behavior in the dataset is fundamentally non-linear, with t-SNE and UMAP providing clearer insights than linear PCA. While six meaningful customer segments exist, the lack of sharp boundaries suggests that customers may share characteristics across segments, requiring flexible and adaptive marketing strategies.

Anomaly detection further refined these insights by identifying customers who may be data errors or may represent strategically important groups. VIP customers should be retained through exclusive services and personalized experiences. Young, high-spending shoppers could be leveraged as brand influencers to attract similar demographics. Disengaged customers, particularly inactive seniors with very low spending scores, should be deprioritized in marketing campaigns to optimize budget allocation.

# Team Contributions
- Timothy Tan: Clustering Analysis, Dimensionality Reduction, Report
- Jun Park: Anomaly Detection, Integrated Analysis, Report