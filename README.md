🚀 Customer Segmentation and Churn Prediction


#📌 Project Overview

This project implements a complete machine learning pipeline for **customer segmentation** and **customer churn prediction**. Customers are first grouped into similar segments using clustering algorithms, and then separate classification models are trained for each segment to predict customer churn.

The objective is to improve churn prediction by considering the behavioral differences among customer groups instead of building a single global model.

---

## ✨Features

* Data preprocessing and cleaning
* Customer segmentation using:

  * K-Means Clustering
  * Hierarchical (Agglomerative) Clustering
* Cluster evaluation using Silhouette Score
* Segment-wise churn prediction
* Hyperparameter tuning using GridSearchCV
* Performance evaluation using multiple metrics

---

##📁 Dataset

The project uses the **Customer Churn Dataset** containing customer demographic information, account details, services subscribed, and churn status.

Example attributes include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Contract
* Payment Method
* Monthly Charges
* Total Charges
* Churn (Target Variable)

---

## 🔄Project Workflow

```
Load Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Label Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Customer Segmentation
 ├── K-Means
 └── Hierarchical Clustering
      │
      ▼
Silhouette Score Evaluation
      │
      ▼
Select K-Means Segments
      │
      ▼
Build Individual Random Forest Models
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
```

---

##🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 🤖Machine Learning Algorithms

### Clustering

* K-Means Clustering
* Agglomerative Hierarchical Clustering

### Classification

* Random Forest Classifier
* Logistic Regression
* Decision Tree Classifier

### Hyperparameter Optimization

* GridSearchCV

---

## Evaluation Metrics

The following metrics are used to evaluate the models:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Silhouette Score (for clustering)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-segmentation-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-segmentation-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```text
pandas
numpy
matplotlib
scikit-learn
```

or install manually:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Running the Project

Run the Python script:

```bash
python customer_segmentation.py
```

or execute the notebook:

```bash
jupyter notebook
```

---

## Output

The program produces:

* Dataset information
* K-Means Silhouette Score
* Hierarchical Clustering Silhouette Score
* Customer segment distribution
* Best hyperparameters for each segment
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Final performance summary

---

## Project Structure

```
Customer-Segmentation-Churn-Prediction/
│
├── customer_segmentation.py
├── customer_churn.csv
├── requirements.txt
├── README.md
└── outputs/
```

---

## Future Improvements

* One-Hot Encoding for categorical variables
* Automatic selection of the optimal number of clusters
* SMOTE for handling class imbalance
* XGBoost and LightGBM models
* Feature importance visualization
* SHAP explainability
* Model deployment using Flask or FastAPI
* Interactive dashboard using Streamlit

---

## Learning Outcomes

This project demonstrates:

* Data preprocessing techniques
* Unsupervised learning using clustering
* Customer segmentation strategies
* Supervised learning for churn prediction
* Hyperparameter tuning
* Model evaluation and comparison
* End-to-end machine learning workflow

---

## License

This project is available under the MIT License.

