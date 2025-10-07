# Customer Segmentation Analysis Using K-Means Clustering

**Data-Driven Marketing Intelligence for Retail**

---

## Overview

This project presents a comprehensive customer segmentation analysis for a supermarket chain, analyzing 33,000 customer records to identify distinct customer groups and develop targeted marketing strategies.

**Author:** Romin Parekh  
**Institution:** UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence  
**Repository:** https://github.com/rominparekh/AIML_PAA_Capstone

---

## Main Report

**📄 [Customer_Segmentation_Academic_Report.pdf](Customer_Segmentation_Academic_Report.pdf)**

This comprehensive academic report (3.8 MB, ~20-25 pages) contains the complete analysis including:

- **Executive Summary** - Key findings and business impact
- **Exploratory Data Analysis** - Statistical insights from 33,000 customer records
- **K-Means Clustering Implementation** - 4 distinct customer segments identified
- **Segment Profiles** - Detailed characteristics and marketing strategies
- **Business Impact** - ROI projections and implementation roadmap
- **Visualizations** - 18 professional plots (12 EDA + 6 Clustering)

---

## Key Findings

### 4 Customer Segments Identified

1. **Affluent Professionals (29.3%)** - High income, educated, management roles
2. **Middle-Aged Value Seekers (25.2%)** - Moderate income, quality-focused
3. **Mature Premium Customers (18.9%)** - Older, affluent, convenience-oriented
4. **Young Budget-Conscious Families (26.7%)** - Younger, price-sensitive, digital-savvy

### Business Impact

- **Revenue Growth:** 5-10% increase projected (Year 1)
- **Marketing Efficiency:** +15-25% conversion rate improvement
- **Customer Lifetime Value:** +20-30% growth
- **ROI:** 13:1 return on segmentation investment

---

## Project Structure

```
├── Customer_Segmentation_Academic_Report.pdf    # Main report (READ THIS)
├── Complete_EDA_Analysis.ipynb                  # Interactive analysis notebook
├── complete_eda_analysis.py                     # EDA automation script
├── customer_clustering_implementation.py        # Clustering script
├── generate_academic_report.py                  # PDF report generator
├── data/
│   └── segmentation_data_33k.csv               # Dataset (33,000 customers)
├── figs/                                        # EDA visualizations (12 plots)
├── output/
│   ├── customer_segments.csv                    # Customers with cluster assignments
│   └── cluster_profiles.csv                     # Segment profiles
└── requirements.txt                             # Python dependencies
```

---

## Quick Start

### 1. Read the Report
```bash
open Customer_Segmentation_Academic_Report.pdf
```

### 2. Run Interactive Analysis
```bash
jupyter lab Complete_EDA_Analysis.ipynb
```

The notebook includes:
- **Part 1:** Complete EDA (12 visualizations)
- **Part 2:** K-Means Clustering (6 visualizations)
- Segment profiling and business insights

### 3. Regenerate Outputs

**EDA only:**
```bash
python complete_eda_analysis.py
```

**Clustering only:**
```bash
python customer_clustering_implementation.py
```

**PDF Report:**
```bash
python generate_academic_report.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- reportlab >= 3.6.0 (for PDF generation)

---

## Dataset

- **Size:** 33,000 customer records
- **Features:** 8 (ID, Sex, Marital status, Age, Education, Income, Occupation, Settlement size)
- **Quality:** 0% missing values, 0 duplicates, 100% complete
- **Source:** Supermarket customer database with demographic and socioeconomic attributes

---

## Methodology

### Exploratory Data Analysis
- Univariate, bivariate, and multivariate analysis
- Statistical validation (ANOVA, Kruskal-Wallis tests)
- Grouped analysis (Age×Education, Sex×Education)
- 12 professional visualizations

### K-Means Clustering
- Optimal K selection using 4 metrics (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Feature engineering (StandardScaler + One-Hot Encoding)
- K=4 determined as optimal
- PCA visualization for cluster validation
- 6 clustering visualizations

---

## Results

### Statistical Validation
All segmentation variables show highly significant effects on income (p < 0.001):
- Education: F = 527.66
- Occupation: F = 9,544.17
- Settlement size: F = 4,066.76

### Cluster Quality Metrics
- Silhouette Score: 0.445 (good separation)
- Davies-Bouldin Index: 0.891 (low overlap)
- Calinski-Harabasz Index: 14,276 (strong separation)

---

## Contact

**Romin Parekh**  
Email: rominparekh@gmail.com  
GitHub: https://github.com/rominparekh/AIML_PAA_Capstone

---

## License

This project is part of an academic portfolio. Data has been anonymized for privacy.

---

**Last Updated:** October 2025

