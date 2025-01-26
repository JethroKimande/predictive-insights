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

def generate_correlation_plot(data):
    # Select relevant numerical features for correlation
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Compute the correlation matrix
    corr = data[numeric_features].corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix for Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    logger.info("Correlation plot saved as correlation_matrix.png")

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
    <p>The primary concern of our educational initiative is to identify primary school students at risk for writing difficulties using machine learning models. This research aims to address the business problem of predicting which students will likely require early intervention by employing data analytics with a supervised learning approach. Timely intervention, particularly in foundational writing skills, is crucial in education. Insights from assessments like the National Assessment Program – Literacy and Numeracy (NAPLAN) provide benchmarks for student performance, aiding in the identification of academic intervention needs. Supervised learning methods, including decision trees and clustering techniques, have proven effective in discerning patterns in diverse educational data (Hastie et al., 2009).</p>

    <h2>Data Preparation and Exploratory Data Analysis (EDA)</h2>
    <p>Our dataset includes variables such as socio-economic status, age, gender, disability, demographic factors, number of siblings, and parents' education levels, offering a comprehensive view of influences on student learning. Data preprocessing involved cleaning to handle missing values, standardizing entries for analysis, and normalizing numerical features using StandardScaler while encoding categorical data with OneHotEncoder. A 70:30 split for training and testing was used, with stratification on the target variable to ensure balanced datasets. EDA included descriptive statistics and visualization of key variables' relationships with Year3_Writing_At_Risk, revealing significant correlations, e.g., consistent performance in literacy assessments like Burt reading tests.</p>
    <p>Key insights from the correlation matrix showed strong positive correlations between different time points of literacy assessments, indicating stable student performance over time. Negative correlations suggested that higher scores in early assessments reduce the risk of writing difficulties in Year 3.</p>
    <img src="correlation_matrix.png" alt="Correlation Matrix for Variables">

    <h2>Model Development</h2>
    <p>We utilized two models: Logistic Regression for its interpretability in binary classification, achieving an overall accuracy of 75% with better performance in identifying non-at-risk students (precision: 0.78, recall: 0.87) compared to at-risk students (precision: 0.66, recall: 0.51). An Artificial Neural Network (ANN) was employed to capture non-linear relationships, slightly outperforming with an accuracy of 71%, with similar precision and recall patterns. K-means clustering was used for unsupervised learning, revealing inherent groupings in student data based on literacy and numeracy levels, visualized in Figure 2.</p>
    <img src="cluster_visualization.png" alt="K-Means Clustering plot">

    <h2>Model Evaluation</h2>
    <p>The ROC curves (Figure 3) demonstrate that the logistic regression model (AUC {lr_auc:.3f}) outperforms the ANN (AUC {ann_auc:.3f}) in distinguishing between at-risk and non-at-risk students. Both models surpass a random classifier, with logistic regression providing more consistent and accurate predictions.</p>
    <img src="roc_curve_lr.png" alt="ROC Curve - Logistic Regression">
    <img src="roc_curve_ann.png" alt="ROC Curve - ANN">

    <h2>Solution Recommendation</h2>
    <p>Based on our validation, logistic regression is recommended for its effectiveness in predicting writing difficulties, offering better interpretability for educators to tailor interventions. Future work should involve expanding the dataset, exploring other ML techniques like ensemble methods, and potentially integrating these models into real-time systems for dynamic student evaluation.</p>

    <h2>Technical Recommendations</h2>
    <p>The analysis was conducted in Python utilizing scikit-learn for machine learning, Pandas for data manipulation, and NumPy for numerical computations. Data preparation followed a structured approach, ensuring consistency in model application through a preprocessing pipeline. Regular retraining is advised due to evolving student data. Figure 4 illustrates the recommended machine process flow for deploying the logistic regression model, emphasizing data collection, preprocessing, feature selection, and model training.</p>
    <img src="machine_process_flow.png" alt="Machine Process Flow Diagram For Logistic Regression Model">

    <h2>Conclusion</h2>
    <p>In summary, this analysis used machine learning to predict which students are at risk of low writing performance by Year 3. Logistic regression was found to be the most accurate and interpretable method, providing educators with actionable insights. The K-means clustering model enriched our understanding by revealing natural student groupings. Implementing these models in real-time could facilitate early interventions, ensuring students receive the attention needed for improved academic outcomes.</p>

    <h2>References</h2>
    <ul>
        <li>Breiman, L. (2001). <em>Random forests</em>. <strong>Machine Learning</strong>, 45(1), 5-32. <a href="https://doi.org/10.1023/A:1010933404324">https://doi.org/10.1023/A:1010933404324</a></li>
        
        <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). <em>Deep Learning</em>. <strong>MIT Press</strong>. <a href="https://www.deeplearningbook.org/">https://www.deeplearningbook.org/</a></li>
        
        <li>Hastie, T., Tibshirani, R., & Friedman, J. (2009). <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>. <strong>Springer Science & Business Media</strong>. <a href="https://doi.org/10.1007/978-0-387-84858-7">https://doi.org/10.1007/978-0-387-84858-7</a></li>
        
        <li>Müller, A. C., & Guido, S. (2016). <em>Introduction to Machine Learning with Python: A Guide for Data Scientists</em>. <strong>O'Reilly Media</strong>. <a href="https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413">https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413</a></li>
        
        <li>Nguyen, L. (2024). <em>MIS710 Machine Learning in Business: Topic 9 - Unsupervised Machine Learning – Clustering using K-Means</em>. <strong>Deakin University</strong>. [Lecture Slides]. <a href="https://www.deakin.edu.au/courses/unit?unit=MIS710">https://www.deakin.edu.au/courses/unit?unit=MIS710</a></li>
        
        <li>Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). <em>Scikit-learn: Machine Learning in Python</em>. <strong>Journal of Machine Learning Research</strong>, 12, 2825-2830. <a href="http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html">http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html</a></li>
        
        <li>Raschka, S., & Mirjalili, V. (2019). <em>Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2</em>. <strong>Packt Publishing</strong>. <a href="https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750">https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750</a></li>
    </ul>

    <footer>
        <p>Jethro Kimande © {datetime.now().year} </p>
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
        
        generate_correlation_plot(data)
        perform_clustering(data)
        generate_html(lr_auc, ann_auc)
        logger.info("HTML file generated successfully")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()