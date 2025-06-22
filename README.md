# Job Fit AI Resume Scanner

A high-speed, AI-powered resume screening tool that classifies resumes into 25+ job roles using NLP and K-Nearest Neighbors (KNN). Achieved **99.5% accuracy** with a runtime of **0.1 seconds per resume**, reducing manual screening time by **98.33%** and improving hiring efficiency and consistency.

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
- [Authors](#authors)

---

## Problem Statement

- 30%–40% of applicants apply to the wrong roles.
- Manual resume reviews are slow, inconsistent, and error-prone.
- High application volume overwhelms HR teams.

---

## Project Goals

- Automate resume classification using machine learning.
- Achieve high accuracy with low runtime.
- Enhance consistency in candidate evaluation.
- Reduce the time and effort recruiters spend on screening.

---

## Dataset Overview

- **Source**: Public CSV Resume Dataset from Kaggle
- **Size**: ~1,000 resumes across 25+ job categories
- **Structure**: Each resume is paired with a labeled job role
- **Limitation**: Some roles overrepresented (e.g., Software Engineer)

---

## Data Cleaning & Preprocessing

**Stage 1 – Regex Cleaning**:

- Removed URLs (LinkedIn, portfolios)
- Removed hashtags (#OpenToWork), mentions (@user)
- Removed punctuation, emojis, and non-English characters
- Normalized whitespace

**Stage 2 – Stopword Filtering**:

- Used `CountVectorizer` to ignore uninformative words (e.g., "the", "or", "is")

---

## Exploratory Data Analysis (EDA)

### Resume Count by Category

![Resume Volume](images/resume_volume.png)

### Resume Length by Role

![Resume Lengths](images/resume_length.png)

### Word Clouds

![Word Cloud](images/wordcloud.png)

### TF-IDF Feature Distributions

**Data Science Word Counts**

![TF-IDF](../images/tfidf_matrix_Data Science.png)

**Sales Word Counts**

![TF-IDF](../images/tfidf_matrix_Sales.png)

- There is a clear difference between the keywords in a Data Science/Technical resume compared to a Sales resume

---

## Modeling Approach

We tested multiple ML algorithms and selected **KNN** for its:

- **Speed**: Classifies in ~0.1 seconds
- **Simplicity**: Memory-based, no complex training
- **Accuracy**: 99.5% on test set
- **Scalability**: Works well with clean, high-dimensional TF-IDF vectors

**KNN Settings**:

- `k = 3` (chosen via elbow plot)
- Distance metric: Euclidean

### Elbow Plot (Optimal k)

![Elbow Plot](images/elbow_plot.png)

---

## Results & Evaluation

| Metric      | Value                       |
| ----------- | --------------------------- |
| Accuracy    | 99.5%                       |
| Speed       | 0.1 sec / resume            |
| Method      | TF-IDF + KNN                |
| Consistency | High across most categories |

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### TSNE Clustering (Resume Embedding Space)

![TSNE](images/tsne.png)

---

## Business Impact

- Reduced manual screening time by **98.33%**
- Improved consistency and reduced bias in evaluation
- Scalable across departments and roles
- Enables data-driven, faster hiring decisions

---

## Limitations

- Imbalanced class distribution across roles
- Dataset may differ from real-world resume formats
- May miss subtle distinctions between similar roles

---

## Next Steps

- Integrate into HR systems or ATS
- Retrain with real, current applicant resumes
- Implement semantic matching (BERT/GPT)
- Suggest resume optimizations or job fit recommendations

---

## Authors

Built by Aditya Sharma as part of a collaborative team project.
Project contributors: Ayan Bhakta, Vidhit Dureja, Hridhay Prabahar, & Aditya Sharma
