# Resume Classification Using Machine Learning

This project is focused on automating the classification of resumes into specific job categories using Natural Language Processing (NLP) and supervised machine learning techniques. It uses TF-IDF for feature extraction and multiple classifiers for evaluating performance.

## 📁 Dataset

- **File Used**: `UpdatedResumeDataSet.csv`
- **Columns**:
  - `Category`: The target variable representing job roles.
  - `Resume`: The textual content of resumes.

## ⚙️ Project Workflow

1. **Data Exploration**
   - Read and inspect dataset shape, sample rows, and category distribution.
   - Visualize the original and balanced category distributions using bar plots and pie charts.

2. **Data Preprocessing**
   - Cleaned the resume text using regular expressions (removing links, punctuation, special characters, etc.).
   - Encoded the `Category` column using `LabelEncoder`.

3. **Balancing the Dataset**
   - Used oversampling to balance all categories to the same sample size (max category count).

4. **Text Vectorization**
   - Converted resume text into TF-IDF vectors using `TfidfVectorizer`.

5. **Train-Test Split**
   - Split the data into training and testing sets with an 80-20 ratio.

6. **Model Training & Evaluation**
   - Trained multiple models including:
     - `KNeighborsClassifier`
     - `Support Vector Classifier (SVC)`
   - Used `OneVsRestClassifier` for multi-class handling.
   - Achieved **100% accuracy** due to oversampling and possible data leakage.

7. **Metrics Used**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)

## 📊 Visualization

- `Seaborn` used for count plots.
- `Matplotlib` used for pie chart visualization of class distribution.

## 🚀 Requirements

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
