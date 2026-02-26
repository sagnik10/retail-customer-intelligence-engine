# Retail Customer Intelligence Engine

An automated end-to-end retail analytics system that performs structural modeling, customer segmentation, anomaly detection, feature importance ranking, similarity modeling, and executive-grade PDF reporting.

---

## Overview

Retail Customer Intelligence Engine is a production-ready analytics pipeline designed to extract behavioral intelligence from retail customer datasets.  

The system automatically:

- Detects and preprocesses numeric features  
- Performs dimensionality reduction (PCA)  
- Identifies anomalies using Isolation Forest  
- Segments customers using KMeans clustering  
- Ranks feature importance using Mutual Information  
- Generates similarity-based customer recommendations  
- Produces professional PDF intelligence reports  

This project is designed for scalable analytics in Kaggle environments as well as local execution.

---

## Core Capabilities

### Structural Modeling
- Standardization of features
- Principal Component Analysis (PCA)
- Explained variance analysis
- PCA projection visualization
- PCA loading heatmaps

### Anomaly Detection
- Isolation Forest-based outlier detection
- Anomaly score distribution analysis
- Behavioral anomaly visualization

### Customer Segmentation
- KMeans clustering
- Elbow method analysis
- Silhouette score evaluation
- Cluster distribution reporting
- Cluster projection visualization

### Statistical & Distribution Analysis
- Target distribution histogram
- Boxplot visualization
- Q-Q normality plot
- Rolling mean & volatility
- Z-score anomaly bands
- Cumulative growth index
- Correlation matrix heatmap
- ACF & PACF time-dependence analysis
- Fourier frequency spectrum analysis

### Feature Intelligence
- Mutual Information feature importance ranking
- Horizontal importance visualization

### Similarity Modeling
- Cosine similarity using Nearest Neighbors
- Top-N customer recommendation mapping
- Memory-safe implementation (no NxN explosion)

### Reporting
- Fully automated PDF report generation
- Executive summary
- Chart-by-chart visualization embedding
- Structured intelligence narrative

---

## Project Structure

```
Retail_Customer_Intelligence_Engine/
│
├── retail_customer_segmentation.csv
├── analyser.py
├── Output/
│   ├── charts/
│   ├── models/
│   ├── recommendations/
│   └── Intelligence_Report.pdf
│
└── README.md
```

---

## Installation

### Required Libraries

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels reportlab
```

---

## How to Run (Local)

1. Place your CSV file in the project root.
2. Update file path inside the script if necessary.
3. Run:

```
python analyser.py
```

Outputs will be generated in:

```
Output/
```

---

## How to Run (Kaggle)

- Upload the notebook or script
- Ensure dataset is available under `/kaggle/input`
- Outputs will be written to:

```
/kaggle/working/Output
```

---

## Dataset Requirements

- At least two numeric columns
- Preferably retail behavioral features such as:
  - Age
  - Income
  - Purchase Frequency
  - Average Monthly Spend
  - Order Value
  - Engagement metrics
- ID columns are automatically excluded from modeling

---

## Output Artifacts

### Charts
- PCA Variance
- PCA Projection
- PCA Loadings
- Anomaly Detection
- Anomaly Score Distribution
- KMeans Elbow
- Cluster Projection
- Cluster Distribution
- Rolling Mean
- Rolling Volatility
- Rolling Z-Score
- Cumulative Growth
- Correlation Heatmap
- Distribution Histogram
- Boxplot
- Q-Q Plot
- ACF
- PACF
- Feature Importance

### Models
- scaler.pkl
- pca.pkl

### Recommendations
- entity_similarity_recommendations.csv

### Report
- Intelligence_Report.pdf

---

## Technical Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Statsmodels
- ReportLab

---

## Use Cases

- Retail customer segmentation
- Behavioral analytics
- Revenue pattern modeling
- Anomaly detection
- Executive reporting automation
- Feature impact analysis
- Similarity-based targeting

---

## Scalability Notes

- Uses StandardScaler + PCA for dimensional compression
- Uses NearestNeighbors instead of full cosine matrix to avoid memory overflow
- Works efficiently on 50k+ customer datasets
- Designed for Kaggle and production environments

---

## Example Applications

- High-value customer identification
- Discount-sensitive segmentation
- Behavioral risk detection
- Revenue driver analysis
- Targeted marketing optimization
- Customer similarity-based recommendation systems

---

## Future Enhancements

- Segment persona labeling
- Revenue contribution modeling
- Churn probability modeling
- SHAP-based explainability
- Dashboard integration
- API deployment

---

## Author

Sagnik (Retail Analytics & Intelligence Systems)

---

## License

MIT License
