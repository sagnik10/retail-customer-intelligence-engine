import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram, detrend
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import warnings
warnings.filterwarnings("ignore")

start=time.time()

BASE_DIR=os.getcwd()
INPUT_FILE=os.path.join(BASE_DIR,"retail_customer_segmentation.csv")
OUTPUT_DIR=os.path.join(BASE_DIR,"Output")
CHART_DIR=os.path.join(OUTPUT_DIR,"charts")
MODEL_DIR=os.path.join(OUTPUT_DIR,"models")
REC_DIR=os.path.join(OUTPUT_DIR,"recommendations")

os.makedirs(CHART_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)
os.makedirs(REC_DIR,exist_ok=True)

DARK="#0b1220"
PANEL="#111827"
TEXT="#e5e7eb"
ACCENT="#22d3ee"
ACCENT2="#a78bfa"
ACCENT3="#34d399"
ACCENT4="#f59e0b"

plt.rcParams.update({
"figure.facecolor":DARK,
"axes.facecolor":PANEL,
"text.color":TEXT,
"axes.labelcolor":TEXT,
"xtick.color":TEXT,
"ytick.color":TEXT,
"font.size":13
})

df=pd.read_csv(INPUT_FILE)
df.columns=[c.lower().replace(" ","_") for c in df.columns]

id_cols=[c for c in df.columns if "id" in c]
numeric=df.select_dtypes(include=np.number).columns.tolist()
numeric=[c for c in numeric if c not in id_cols]

target="avg_monthly_spend"

df[numeric]=df[numeric].fillna(df[numeric].median())

mean_val=round(df[target].mean(),2)
std_val=round(df[target].std(),2)
min_val=round(df[target].min(),2)
max_val=round(df[target].max(),2)

scaler=StandardScaler()
scaled=scaler.fit_transform(df[numeric])
pickle.dump(scaler,open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb"))

pca=PCA(n_components=min(6,len(numeric)))
pca_data=pca.fit_transform(scaled)
pickle.dump(pca,open(os.path.join(MODEL_DIR,"pca.pkl"),"wb"))

explained_var=round(np.sum(pca.explained_variance_ratio_)*100,2)

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(range(1,len(pca.explained_variance_ratio_)+1),
        np.cumsum(pca.explained_variance_ratio_),
        color=ACCENT,linewidth=3)
ax.set_title("Principal Component Variance Explained")
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Cumulative Explained Variance Ratio")
fig.savefig(os.path.join(CHART_DIR,"pca_variance.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(pca_data[:,0],pca_data[:,1],s=20,color=ACCENT3)
ax.set_title("PCA Projection of Customers")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.savefig(os.path.join(CHART_DIR,"pca_projection.png"),dpi=300,bbox_inches="tight")
plt.close()

iso=IsolationForest(contamination=0.03,random_state=42)
anom=iso.fit_predict(scaled)
anomaly_count=int((anom==-1).sum())

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(df[target],anom,c=anom,cmap="coolwarm",s=20)
ax.set_title("Spending Anomaly Detection")
ax.set_xlabel("Average Monthly Spend")
ax.set_ylabel("Anomaly Flag")
fig.savefig(os.path.join(CHART_DIR,"anomaly.png"),dpi=300,bbox_inches="tight")
plt.close()

kmeans=KMeans(n_clusters=4,n_init=20,random_state=42)
clusters=kmeans.fit_predict(scaled)
df["cluster"]=clusters
sil=round(silhouette_score(scaled,clusters),3)

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(pca_data[:,0],pca_data[:,1],c=clusters,cmap="viridis",s=20)
ax.set_title("Customer Segmentation Clustering")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.savefig(os.path.join(CHART_DIR,"cluster.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
df["cluster"].value_counts().plot(kind="bar",color=ACCENT3,ax=ax)
ax.set_title("Cluster Distribution")
ax.set_xlabel("Cluster")
ax.set_ylabel("Customer Count")
fig.savefig(os.path.join(CHART_DIR,"cluster_distribution.png"),dpi=300,bbox_inches="tight")
plt.close()

corr=df[numeric].corr()
fig,ax=plt.subplots(figsize=(14,10))
sns.heatmap(corr,cmap="viridis",ax=ax)
ax.set_title("Feature Correlation Matrix")
fig.savefig(os.path.join(CHART_DIR,"correlation.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
sns.histplot(df[target],bins=40,color=ACCENT,ax=ax)
ax.set_title("Spending Distribution")
ax.set_xlabel("Average Monthly Spend")
ax.set_ylabel("Frequency")
fig.savefig(os.path.join(CHART_DIR,"distribution.png"),dpi=300,bbox_inches="tight")
plt.close()

mi=mutual_info_regression(df[numeric],df[target])
imp=pd.Series(mi,index=numeric).sort_values()

fig,ax=plt.subplots(figsize=(14,8))
imp.plot(kind="barh",color=ACCENT2,ax=ax)
ax.set_title("Feature Importance for Monthly Spend")
ax.set_xlabel("Mutual Information Score")
fig.savefig(os.path.join(CHART_DIR,"feature_importance.png"),dpi=300,bbox_inches="tight")
plt.close()

nn=NearestNeighbors(n_neighbors=6,metric="cosine",algorithm="brute")
nn.fit(scaled)
distances,indices=nn.kneighbors(scaled)

recommendations=[]
for i in range(len(df)):
    for j in range(1,6):
        recommendations.append({
            "source_index":i,
            "recommended_index":indices[i][j],
            "similarity":round(1-distances[i][j],4)
        })

rec_df=pd.DataFrame(recommendations)
rec_df.to_csv(os.path.join(REC_DIR,"customer_similarity_recommendations.csv"),index=False)

execution=round(time.time()-start,2)

styles=getSampleStyleSheet()

title_style=ParagraphStyle(name="title",fontSize=34,leading=42,alignment=1,
                           textColor=HexColor("#22d3ee"),spaceAfter=50,spaceBefore=50)

heading_style=ParagraphStyle(name="heading",fontSize=22,leading=28,alignment=1,
                             textColor=HexColor("#a78bfa"),spaceBefore=40,spaceAfter=30)

body_style=ParagraphStyle(name="body",fontSize=12,leading=20,spaceAfter=40)

doc=SimpleDocTemplate(os.path.join(OUTPUT_DIR,"Retail_Customer_Intelligence_Report.pdf"),
                      leftMargin=72,rightMargin=72,topMargin=72,bottomMargin=72)

elements=[]
elements.append(Paragraph("Retail Customer Intelligence Report",title_style))

summary=f"""
Executive Summary<br/><br/>
Dataset Size: {len(df)} customers<br/>
Target Variable: Average Monthly Spend<br/>
Mean Spend: {mean_val}<br/>
Standard Deviation: {std_val}<br/>
Min Spend: {min_val}<br/>
Max Spend: {max_val}<br/>
Cluster Quality Score: {sil}<br/>
Detected Outliers: {anomaly_count}<br/>
PCA Explained Variance: {explained_var}%<br/>
Execution Time: {execution} seconds<br/><br/>
This report provides structural intelligence, anomaly detection,
segmentation modeling, feature importance analysis, and customer similarity mapping.
"""

elements.append(Paragraph(summary,body_style))
elements.append(PageBreak())

chart_explanations={
"pca_variance.png":"This chart shows how much structural variance is captured by successive principal components.",
"pca_projection.png":"This projection visualizes customer distribution in reduced dimensional space.",
"anomaly.png":"Anomaly detection identifies unusual spending behaviour relative to the population.",
"cluster.png":"Clustering separates customers into behaviorally distinct groups.",
"cluster_distribution.png":"This shows how customers are distributed across segments.",
"correlation.png":"Correlation matrix reveals interdependencies among retail behavioural features.",
"distribution.png":"Distribution of monthly spending shows spread and skewness of revenue patterns.",
"feature_importance.png":"Mutual information ranking identifies strongest drivers of monthly spend."
}

charts=sorted(os.listdir(CHART_DIR))

for chart in charts:
    elements.append(Paragraph(chart.replace("_"," ").replace(".png","").title(),heading_style))
    elements.append(Image(os.path.join(CHART_DIR,chart),
                          width=6.5*inch,height=4.5*inch))
    elements.append(Spacer(1,40))
    elements.append(Paragraph(chart_explanations.get(chart,
                    "Analytical visualization of retail behavioural intelligence."),body_style))
    elements.append(PageBreak())

doc.build(elements)

print("Complete")
print("Execution Time:",execution)