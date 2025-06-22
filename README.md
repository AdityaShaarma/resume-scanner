# Job Fit AI Resume Scanner

A high-speed, AI-powered resume screening tool that classifies resumes into 25+ job roles using NLP and K-Nearest Neighbors (KNN). Achieved **99.5% accuracy** with a runtime of **0.1 seconds per resume**, reducing manual screening time by **98.33%** and improving hiring efficiency and consistency.

Dataset: [Updated Resume Dataset on Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Dataset Overview](#dataset-overview)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling Approach](#modeling-approach)
- [Results & Evaluation](#results--evaluation)
- [Business Impact](#business-impact)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Libraries Used](#libraries-used)
- [Acknowledgments](#acknowledgments)
- [Authors](#authors)

---

## Problem Statement

- 30%–40% of applicants apply to the wrong roles
- Manual resume reviews are slow, inconsistent, and error-prone
- High application volume overwhelms HR teams

---

## Project Goals

- Automate resume classification using machine learning
- Achieve high accuracy with low runtime
- Enhance consistency in candidate evaluation
- Reduce time and effort recruiters spend on screening

---

## Dataset Overview

- **Source**: [Updated Resume Dataset on Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)
- **Size**: ~1,000 resumes across 25+ job categories
- **Structure**: Each resume is paired with a labeled job role
- **Limitation**: Some roles are overrepresented (e.g., Software Engineer)

---

## Data Cleaning & Preprocessing

### Stage 1 – Regex Cleaning

- Removed URLs, email addresses, and HTML tags
- Filtered out hashtags and @mentions
- Removed punctuation, emojis, and non-English characters
- Normalized whitespace and casing

### Stage 2 – Stopword Filtering

- Removed common stopwords using `CountVectorizer`
- Tokenized resumes for downstream feature extraction

---

## Exploratory Data Analysis (EDA)

### Resume Count by Category

![Resume Volume](images/resume_volume.png)

### Resume Length by Role

![Resume Length](images/resume_length.png)

### Word Clouds (All Roles)

![Word Cloud](images/wordcloud.png)

### TF-IDF Keyword Frequencies

**Data Science**

![TF-IDF Data Science](images/tfidf_matrix_Data Science.png)

**Sales**

![TF-IDF Sales](images/tfidf_matrix_Sales.png)

> Different roles display unique keyword distributions—Sales resumes focus on communication, while Data Science resumes emphasize tools and methods.

---

## Modeling Approach

We evaluated multiple ML algorithms including Logistic Regression, SVM, and Random Forest. **K-Nearest Neighbors (KNN)** was selected for its:

- **Speed**: Classifies in ~0.1 seconds
- **Simplicity**: Memory-based, no iterative training
- **Accuracy**: 99.5% on test set
- **Scalability**: Performs well on TF-IDF feature space

### KNN Parameters

- `k = 3` (chosen via elbow method)
- Distance metric: Euclidean

### Elbow Plot (Optimal k)

![Elbow Plot](images/elbow_plot.png)

---

## Results & Evaluation

| Metric      | Value                  |
| ----------- | ---------------------- |
| Accuracy    | 99.5%                  |
| Speed       | 0.1 seconds per resume |
| Model       | K-Nearest Neighbors    |
| Consistency | High across categories |

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### TSNE Clustering

![TSNE](images/tsne.png)

> t-SNE visualization confirms that resumes are clearly separable in embedded space.

---

## Business Impact

- Reduced screening time by 98.33%
- Improved consistency and reduced hiring bias
- Scalable across departments and job types
- Enabled data-driven talent acquisition decisions

---

## Limitations

- Imbalanced class distribution (some roles underrepresented)
- Dataset lacks real-world resume diversity (e.g., formatting, multi-language)
- Does not currently understand nuanced roles (e.g., hybrid tech + sales)

---

## Next Steps

- Integrate with HR systems (e.g., Workday, Greenhouse)
- Retrain on live applicant resumes
- Incorporate deep learning (e.g., BERT/GPT for semantic analysis)
- Add resume feedback or optimization recommendations

---

## Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
```

---

## Acknowledgments

- Dataset sourced from [Kaggle: Updated Resume Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)

- Thanks to the **University of Illinois Urbana-Champaign** faculty and classmates for guidance and constructive feedback during development

---

## Authors

Project developed by **Aditya Sharma**

Collaborators:

- Ayan Bhakta
- Vidhit Dureja
- Hridhay Prabahar
