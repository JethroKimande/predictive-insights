import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "LA4PSchools.csv")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path}")

        features = data.drop(columns=['StudentID', 'Year3_Writing_At_Risk'])
        target = data['Year3_Writing_At_Risk']
        
        num_cols = features.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = features.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, stratify=target, random_state=42)
        return X_train, X_test, y_train, y_test, preprocessor, data
    except FileNotFoundError:
        logger.error(f"File {data_path} not found")
        raise
    except Exception as e:
        logger.error(f"An error occurred while preparing data: {e}")
        raise

def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, preprocessor):
    try:
        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        logistic_pipeline.fit(X_train, y_train)
        
        y_pred = logistic_pipeline.predict(X_test)
        logger.info("Logistic Regression performance evaluated")
        print("Logistic Regression Classification Report:")
        print(classification_report(y_test, y_pred))
        
        y_proba = logistic_pipeline.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, y_proba)
        print(f"Logistic Regression AUC: {lr_auc:.3f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_lr.png')
        plt.close()
        
        return logistic_pipeline, lr_auc
    except Exception as e:
        logger.error(f"Error in Logistic Regression training: {e}")
        raise

def train_and_evaluate_ann(X_train, y_train, X_test, y_test, preprocessor):
    try:
        ann_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42))
        ])
        ann_pipeline.fit(X_train, y_train)
        
        y_pred = ann_pipeline.predict(X_test)
        logger.info("ANN performance evaluated")
        print("ANN (MLP) Classification Report:")
        print(classification_report(y_test, y_pred))
        
        y_proba = ann_pipeline.predict_proba(X_test)[:, 1]
        ann_auc = roc_auc_score(y_test, y_proba)
        print(f"ANN AUC: {ann_auc:.3f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ANN (AUC = {ann_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ANN')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_ann.png')
        plt.close()
        
        return ann_pipeline, ann_auc
    except Exception as e:
        logger.error(f"Error in ANN training: {e}")
        raise

def perform_clustering(data):
    try:
        features = data.drop(columns=['StudentID', 'Year3_Writing_At_Risk'])
        
        num_cols = features.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = features.select_dtypes(include=['object']).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        
        clustering_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('kmeans', KMeans(n_clusters=3, random_state=42))
        ])
        
        X_cluster = clustering_pipeline.fit_predict(features)
        data['Cluster'] = X_cluster
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(preprocessor.fit_transform(features))
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_cluster, cmap='plasma', label='Cluster')
        plt.title('K-Means Clustering of Students (PCA Reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar()
        plt.savefig('cluster_visualization.png')
        logger.info("Clustering visualization saved")
        plt.close()
    except Exception as e:
        logger.error(f"Error in clustering process: {e}")
        raise

def generate_html(lr_auc, ann_auc):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Insights: Unveiling Student Writing Risks</title>
    <meta property="og:title" content="Predictive Insights: Unveiling Student Writing Risks">
    <meta name="twitter:title" content="Predictive Insights: Unveiling Student Writing Risks">
    <style>
        body {{font-family: Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 800px; padding: 20px;}}
        img {{max-width: 100%; height: auto; display: block; margin: 0 auto 20px;}}
        h1, h2 {{color: #333;}}
    </style>
</head>
<body>
    <h1>Predictive Insights: Unveiling Student Writing Risks</h1>
    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Introduction</h2>
    <p>Our aim is to predict primary school students at risk for writing difficulties using machine learning...</p>

    <h2>Model Performance</h2>
    <p><strong>Logistic Regression AUC:</strong> {lr_auc:.3f}</p>
    <p><strong>ANN AUC:</strong> {ann_auc:.3f}</p>
    <img src="roc_curve_lr.png" alt="ROC Curve - Logistic Regression">
    <img src="roc_curve_ann.png" alt="ROC Curve - ANN">

    <h2>Clustering Analysis</h2>
    <p>This visualization shows how students are grouped based on various features using K-Means clustering.</p>
    <img src="cluster_visualization.png" alt="Clustering Visualization">

    <h2>Conclusion</h2>
    <p>From our analysis, logistic regression was more effective in predicting students at risk...</p>

    <h2>Technical Recommendations</h2>
    <p>The analysis was done using Python with libraries like scikit-learn, Pandas, and NumPy...</p>

    <footer>
        <p>© {datetime.now().year} Your Name</p>
    </footer>
</body>
</html>
    """
    with open('index.html', 'w') as file:
        file.write(html_content)

def main():
    try:
        X_train, X_test, y_train, y_test, preprocessor, data = load_and_prepare_data()
        
        logistic_model, lr_auc = train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, preprocessor)
        ann_model, ann_auc = train_and_evaluate_ann(X_train, y_train, X_test, y_test, preprocessor)
        
        perform_clustering(data)
        generate_html(lr_auc, ann_auc)
        logger.info("HTML file generated successfully")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()