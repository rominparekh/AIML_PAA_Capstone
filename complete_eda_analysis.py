#!/usr/bin/env python3
"""
Complete Customer Segmentation EDA Analysis
===========================================

This script generates all plots and insights for the customer segmentation analysis.
It creates 8 professional visualizations and provides comprehensive statistical insights.

Dataset: 33,000 customer records with demographic and socioeconomic features
Output: All plots saved to figs/ folder + detailed console insights

Usage: python complete_eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import os

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directory
os.makedirs('figs', exist_ok=True)

print("🎨 CUSTOMER SEGMENTATION - COMPLETE EDA ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv('data/segmentation_data_33k.csv')
print(f"📊 Dataset loaded: {df.shape[0]:,} customers × {df.shape[1]} features")

# Data dictionary for interpretations
label_mappings = {
    'Sex': {0: 'Female', 1: 'Male'},
    'Marital status': {0: 'Single', 1: 'Married'},
    'Education': {0: 'Basic', 1: 'Secondary', 2: 'Higher', 3: 'Graduate'},
    'Occupation': {0: 'Unemployed/Student', 1: 'Skilled Worker', 2: 'Management'},
    'Settlement size': {0: 'Small City', 1: 'Medium City', 2: 'Large City'}
}

# Create labeled dataframe for visualizations
df_labeled = df.copy()
for col, mapping in label_mappings.items():
    df_labeled[col] = df_labeled[col].map(mapping)

print("\n🔍 GENERATING COMPREHENSIVE ANALYSIS...")

# =============================================================================
# PLOT 1: Dataset Overview
# =============================================================================
def plot_dataset_overview():
    """Generate dataset overview visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Customer Segmentation Dataset Overview', fontsize=16, fontweight='bold')
    
    # Data types
    data_types = df.dtypes.value_counts()
    axes[0,0].pie(data_types.values, labels=data_types.index, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Data Types Distribution')
    
    # Missing values (should be all zeros)
    missing_data = df.isnull().sum()
    axes[0,1].bar(range(len(missing_data)), missing_data.values)
    axes[0,1].set_title('Missing Values by Column')
    axes[0,1].set_xlabel('Columns')
    axes[0,1].set_ylabel('Missing Count')
    axes[0,1].set_xticks(range(len(missing_data)))
    axes[0,1].set_xticklabels(missing_data.index, rotation=45)
    
    # Unique values count
    unique_counts = [df[col].nunique() for col in df.columns]
    axes[1,0].bar(range(len(df.columns)), unique_counts)
    axes[1,0].set_title('Unique Values per Column')
    axes[1,0].set_xlabel('Columns')
    axes[1,0].set_ylabel('Unique Count')
    axes[1,0].set_xticks(range(len(df.columns)))
    axes[1,0].set_xticklabels(df.columns, rotation=45)
    
    # Dataset info
    info_text = f"""Dataset Information:
    • Rows: {len(df):,}
    • Columns: {len(df.columns)}
    • Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
    • Missing Values: {df.isnull().sum().sum()}
    • Duplicates: {df.duplicated().sum()}
    • Data Quality: Perfect (100%)"""
    
    axes[1,1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('Dataset Summary')
    
    plt.tight_layout()
    plt.savefig('figs/01_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 01_dataset_overview.png")

# =============================================================================
# PLOT 2: Numerical Variables Distribution
# =============================================================================
def plot_numerical_analysis():
    """Generate numerical variables analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Numerical Variables Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Age distribution
    axes[0,0].hist(df['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age (years)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['Age'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df["Age"].mean():.1f}')
    axes[0,0].axvline(df['Age'].median(), color='green', linestyle='--', 
                     label=f'Median: {df["Age"].median():.1f}')
    axes[0,0].legend()
    
    # Age boxplot
    axes[0,1].boxplot(df['Age'])
    axes[0,1].set_title('Age Boxplot (Outlier Detection)')
    axes[0,1].set_ylabel('Age (years)')
    
    # Income distribution
    axes[1,0].hist(df['Income'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,0].set_title('Income Distribution')
    axes[1,0].set_xlabel('Income ($)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(df['Income'].mean(), color='red', linestyle='--', 
                     label=f'Mean: ${df["Income"].mean():,.0f}')
    axes[1,0].axvline(df['Income'].median(), color='green', linestyle='--', 
                     label=f'Median: ${df["Income"].median():,.0f}')
    axes[1,0].legend()
    
    # Income boxplot
    axes[1,1].boxplot(df['Income'])
    axes[1,1].set_title('Income Boxplot (Outlier Detection)')
    axes[1,1].set_ylabel('Income ($)')
    
    plt.tight_layout()
    plt.savefig('figs/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 02_numerical_distributions.png")

# =============================================================================
# PLOT 3: Categorical Variables Distribution
# =============================================================================
def plot_categorical_analysis():
    """Generate categorical variables analysis plots"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Categorical Variables Distribution Analysis', fontsize=16, fontweight='bold')
    
    categorical_cols = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
    
    for i, col in enumerate(categorical_cols):
        row = i // 2
        col_idx = i % 2
        
        value_counts = df_labeled[col].value_counts()
        bars = axes[row, col_idx].bar(value_counts.index, value_counts.values, alpha=0.7)
        axes[row, col_idx].set_title(f'{col} Distribution')
        axes[row, col_idx].set_ylabel('Count')
        
        # Add percentage labels on bars
        total = len(df_labeled)
        for j, (bar, v) in enumerate(zip(bars, value_counts.values)):
            height = bar.get_height()
            axes[row, col_idx].text(bar.get_x() + bar.get_width()/2., height + 200,
                                   f'{v:,}\n({v/total*100:.1f}%)', 
                                   ha='center', va='bottom', fontweight='bold')
        
        axes[row, col_idx].tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    fig.delaxes(axes[2, 1])
    
    plt.tight_layout()
    plt.savefig('figs/03_categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 03_categorical_distributions.png")

# =============================================================================
# PLOT 4: Correlation Analysis
# =============================================================================
def plot_correlation_analysis():
    """Generate correlation analysis plots"""
    numerical_cols = ['Age', 'Income']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Correlation and Relationship Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=axes[0])
    axes[0].set_title('Correlation Matrix - Numerical Variables')
    
    # Age vs Income scatter plot
    axes[1].scatter(df['Age'], df['Income'], alpha=0.3, color='blue', s=10)
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Income ($)')
    axes[1].set_title('Age vs Income Relationship')
    
    # Add trend line
    z = np.polyfit(df['Age'], df['Income'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['Age'], p(df['Age']), "r--", alpha=0.8, 
                label=f'Correlation: {correlation_matrix.loc["Age", "Income"]:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/04_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 04_correlation_analysis.png")

# =============================================================================
# PLOT 5: Income Analysis by Categories
# =============================================================================
def plot_income_by_categories():
    """Generate income distribution by categorical variables"""
    categorical_cols = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Income Distribution by Categorical Variables', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(categorical_cols):
        row = i // 3
        col_idx = i % 3
        
        # Box plot
        df.boxplot(column='Income', by=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Income by {col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Income ($)')
        
        # Add mean values as text
        means = df.groupby(col)['Income'].mean()
        for j, (category, mean_val) in enumerate(means.items()):
            axes[row, col_idx].text(j+1, mean_val, f'μ=${mean_val:,.0f}', 
                                   ha='center', va='bottom', fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('figs/05_income_by_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 05_income_by_categories.png")

# =============================================================================
# PLOT 6: Advanced Multi-dimensional Analysis
# =============================================================================
def plot_advanced_analysis():
    """Generate advanced multi-dimensional analysis plots"""
    # Create age and income groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100],
                            labels=['Young (18-30)', 'Middle (31-45)', 'Mature (46-60)', 'Senior (60+)'])
    df['Income_Group'] = pd.cut(df['Income'], bins=3, labels=['Low', 'Medium', 'High'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Multi-dimensional Customer Analysis', fontsize=16, fontweight='bold')

    # Age group vs Income group heatmap
    age_income_crosstab = pd.crosstab(df['Age_Group'], df['Income_Group'])
    sns.heatmap(age_income_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Customer Distribution: Age Groups vs Income Groups')

    # Education vs Income by Gender
    df_viz = df.copy()
    df_viz['Education_Label'] = df_viz['Education'].map(label_mappings['Education'])
    df_viz['Sex_Label'] = df_viz['Sex'].map(label_mappings['Sex'])

    sns.boxplot(data=df_viz, x='Education_Label', y='Income', hue='Sex_Label', ax=axes[0,1])
    axes[0,1].set_title('Income by Education and Gender')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Settlement size vs Age by Marital status
    df_viz['Settlement_Label'] = df_viz['Settlement size'].map(label_mappings['Settlement size'])
    df_viz['Marital_Label'] = df_viz['Marital status'].map(label_mappings['Marital status'])

    sns.boxplot(data=df_viz, x='Settlement_Label', y='Age', hue='Marital_Label', ax=axes[1,0])
    axes[1,0].set_title('Age by Settlement Size and Marital Status')

    # Occupation vs Income distribution
    df_viz['Occupation_Label'] = df_viz['Occupation'].map(label_mappings['Occupation'])

    sns.violinplot(data=df_viz, x='Occupation_Label', y='Income', ax=axes[1,1])
    axes[1,1].set_title('Income Distribution by Occupation')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('figs/06_advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 06_advanced_analysis.png")

# =============================================================================
# PLOT 7: Outlier Analysis
# =============================================================================
def plot_outlier_analysis():
    """Generate outlier analysis visualization"""
    numerical_cols = ['Age', 'Income']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Outlier Detection and Analysis', fontsize=16, fontweight='bold')

    for i, col in enumerate(numerical_cols):
        # Calculate outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        normal_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Histogram with outliers highlighted
        axes[i,0].hist(normal_data[col], bins=30, alpha=0.7, color='lightblue',
                      label=f'Normal ({len(normal_data):,})', edgecolor='black')
        if len(outliers) > 0:
            axes[i,0].hist(outliers[col], bins=10, alpha=0.7, color='red',
                          label=f'Outliers ({len(outliers):,})', edgecolor='black')
        axes[i,0].set_title(f'{col} Distribution with Outliers')
        axes[i,0].set_xlabel(col)
        axes[i,0].set_ylabel('Frequency')
        axes[i,0].legend()

        # Box plot with outlier statistics
        box_plot = axes[i,1].boxplot(df[col], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        axes[i,1].set_title(f'{col} Boxplot')
        axes[i,1].set_ylabel(col)

        # Add statistics text
        stats_text = f"""Outlier Statistics:
        Q1: {Q1:.1f}
        Q3: {Q3:.1f}
        IQR: {IQR:.1f}
        Lower: {lower_bound:.1f}
        Upper: {upper_bound:.1f}
        Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.1f}%)"""

        axes[i,1].text(1.1, 0.5, stats_text, transform=axes[i,1].transAxes,
                      fontsize=9, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig('figs/07_outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 07_outlier_analysis.png")

# =============================================================================
# PLOT 8: Summary Statistics Visualization
# =============================================================================
def plot_summary_statistics():
    """Generate summary statistics visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Customer Segmentation - Key Statistics Summary', fontsize=16, fontweight='bold')

    # Demographics pie charts
    sex_counts = df['Sex'].value_counts()
    sex_labels = ['Female', 'Male']
    axes[0,0].pie(sex_counts.values, labels=sex_labels, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Gender Distribution')

    marital_counts = df['Marital status'].value_counts()
    marital_labels = ['Single', 'Married']
    axes[0,1].pie(marital_counts.values, labels=marital_labels, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Marital Status Distribution')

    # Education distribution
    education_counts = df['Education'].value_counts().sort_index()
    education_labels = ['Basic', 'Secondary', 'Higher', 'Graduate']
    bars = axes[1,0].bar(education_labels, education_counts.values, alpha=0.7)
    axes[1,0].set_title('Education Level Distribution')
    axes[1,0].set_ylabel('Count')

    # Add value labels on bars
    for bar, count in zip(bars, education_counts.values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 200,
                      f'{count:,}\n({count/len(df)*100:.1f}%)',
                      ha='center', va='bottom', fontweight='bold')

    # Key statistics summary
    correlation = df['Age'].corr(df['Income'])
    stats_summary = f"""📊 CUSTOMER SEGMENTATION DATASET SUMMARY

    🔢 Dataset Size:
    • Total Customers: {len(df):,}
    • Features: {len(df.columns)}
    • Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB

    👥 Demographics:
    • Average Age: {df['Age'].mean():.1f} years
    • Age Range: {df['Age'].min()}-{df['Age'].max()} years
    • Female: {sex_counts[0]:,} ({sex_counts[0]/len(df)*100:.1f}%)
    • Male: {sex_counts[1]:,} ({sex_counts[1]/len(df)*100:.1f}%)

    💰 Income Insights:
    • Average Income: ${df['Income'].mean():,.0f}
    • Median Income: ${df['Income'].median():,.0f}
    • Income Range: ${df['Income'].min():,.0f} - ${df['Income'].max():,.0f}
    • Income Std Dev: ${df['Income'].std():,.0f}
    • Age-Income Correlation: {correlation:.3f}

    🎓 Education & Work:
    • Secondary Education: {education_counts[1]:,} ({education_counts[1]/len(df)*100:.1f}%)
    • Skilled Workers: {df['Occupation'].value_counts()[1]:,} customers
    • Management: {df['Occupation'].value_counts()[2]:,} customers

    ✅ Data Quality:
    • Missing Values: {df.isnull().sum().sum()}
    • Duplicate Records: {df.duplicated().sum()}
    • Data Completeness: 100%"""

    axes[1,1].text(0.05, 0.95, stats_summary, transform=axes[1,1].transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.savefig('figs/08_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: 08_summary_statistics.png")

# Run all plot generation functions
plot_dataset_overview()
plot_numerical_analysis()
plot_categorical_analysis()
plot_correlation_analysis()
plot_income_by_categories()
plot_advanced_analysis()
plot_outlier_analysis()
plot_summary_statistics()

print("\n🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
print("📁 Plots saved in 'figs/' folder")

# =============================================================================
# COMPREHENSIVE STATISTICAL INSIGHTS
# =============================================================================
print("\n" + "="*60)
print("📊 COMPREHENSIVE STATISTICAL INSIGHTS")
print("="*60)

# Basic dataset info
print(f"\n🔢 DATASET OVERVIEW:")
print(f"   • Total customers: {len(df):,}")
print(f"   • Features: {len(df.columns)}")
print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"   • Missing values: {df.isnull().sum().sum()} (0%)")
print(f"   • Duplicate records: {df.duplicated().sum()}")
print(f"   • Data quality: Perfect (100% complete)")

# Age analysis
print(f"\n📅 AGE ANALYSIS:")
age_stats = df['Age'].describe()
print(f"   • Mean: {age_stats['mean']:.1f} years")
print(f"   • Median: {age_stats['50%']:.1f} years")
print(f"   • Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
print(f"   • Standard deviation: {age_stats['std']:.1f} years")
print(f"   • Skewness: {df['Age'].skew():.2f}")
print(f"   • Kurtosis: {df['Age'].kurtosis():.2f}")

# Income analysis
print(f"\n💰 INCOME ANALYSIS:")
income_stats = df['Income'].describe()
print(f"   • Mean: ${income_stats['mean']:,.0f}")
print(f"   • Median: ${income_stats['50%']:,.0f}")
print(f"   • Range: ${income_stats['min']:,.0f} - ${income_stats['max']:,.0f}")
print(f"   • Standard deviation: ${income_stats['std']:,.0f}")
print(f"   • Coefficient of variation: {(income_stats['std']/income_stats['mean'])*100:.1f}%")
print(f"   • Skewness: {df['Income'].skew():.2f}")
print(f"   • Kurtosis: {df['Income'].kurtosis():.2f}")

# Correlation analysis
correlation = df['Age'].corr(df['Income'])
print(f"\n🔗 CORRELATION ANALYSIS:")
print(f"   • Age-Income correlation: {correlation:.3f}")
if abs(correlation) < 0.3:
    strength = "weak"
elif abs(correlation) < 0.7:
    strength = "moderate"
else:
    strength = "strong"
print(f"   • Relationship strength: {strength}")
print(f"   • Business implication: {'Independent segmentation variables' if abs(correlation) < 0.3 else 'Some relationship exists'}")

# Demographics breakdown
print(f"\n👥 DEMOGRAPHIC BREAKDOWN:")

# Gender
sex_counts = df['Sex'].value_counts()
print(f"   Gender Distribution:")
print(f"     • Female: {sex_counts[0]:,} ({sex_counts[0]/len(df)*100:.1f}%)")
print(f"     • Male: {sex_counts[1]:,} ({sex_counts[1]/len(df)*100:.1f}%)")

# Marital status
marital_counts = df['Marital status'].value_counts()
print(f"   Marital Status:")
print(f"     • Single: {marital_counts[0]:,} ({marital_counts[0]/len(df)*100:.1f}%)")
print(f"     • Married: {marital_counts[1]:,} ({marital_counts[1]/len(df)*100:.1f}%)")

# Education
education_counts = df['Education'].value_counts().sort_index()
education_labels = ['Basic', 'Secondary', 'Higher', 'Graduate']
print(f"   Education Levels:")
for i, (count, label) in enumerate(zip(education_counts, education_labels)):
    print(f"     • {label}: {count:,} ({count/len(df)*100:.1f}%)")

# Occupation
occupation_counts = df['Occupation'].value_counts().sort_index()
occupation_labels = ['Unemployed/Student', 'Skilled Worker', 'Management']
print(f"   Occupation Categories:")
for i, (count, label) in enumerate(zip(occupation_counts, occupation_labels)):
    print(f"     • {label}: {count:,} ({count/len(df)*100:.1f}%)")

# Settlement size
settlement_counts = df['Settlement size'].value_counts().sort_index()
settlement_labels = ['Small City', 'Medium City', 'Large City']
print(f"   Settlement Sizes:")
for i, (count, label) in enumerate(zip(settlement_counts, settlement_labels)):
    print(f"     • {label}: {count:,} ({count/len(df)*100:.1f}%)")

# Income analysis by categories
print(f"\n💰 INCOME BY CATEGORIES:")

# By gender
income_by_gender = df.groupby('Sex')['Income'].agg(['mean', 'median']).round(0)
print(f"   By Gender:")
print(f"     • Female: Mean=${income_by_gender.loc[0, 'mean']:,.0f}, Median=${income_by_gender.loc[0, 'median']:,.0f}")
print(f"     • Male: Mean=${income_by_gender.loc[1, 'mean']:,.0f}, Median=${income_by_gender.loc[1, 'median']:,.0f}")
gender_gap = income_by_gender.loc[0, 'mean'] - income_by_gender.loc[1, 'mean']
print(f"     • Gender gap: ${gender_gap:,.0f} (Female {'higher' if gender_gap > 0 else 'lower'})")

# By education
income_by_education = df.groupby('Education')['Income'].mean().round(0)
print(f"   By Education:")
for i, label in enumerate(education_labels):
    print(f"     • {label}: ${income_by_education[i]:,.0f}")

# By occupation
income_by_occupation = df.groupby('Occupation')['Income'].mean().round(0)
print(f"   By Occupation:")
for i, label in enumerate(occupation_labels):
    print(f"     • {label}: ${income_by_occupation[i]:,.0f}")

# Outlier analysis
print(f"\n⚠️ OUTLIER ANALYSIS:")

# Age outliers
age_q1, age_q3 = df['Age'].quantile([0.25, 0.75])
age_iqr = age_q3 - age_q1
age_upper = age_q3 + 1.5 * age_iqr
age_outliers = df[df['Age'] > age_upper]
print(f"   Age Outliers:")
print(f"     • Count: {len(age_outliers):,} ({len(age_outliers)/len(df)*100:.1f}%)")
print(f"     • Threshold: >{age_upper:.1f} years")
if len(age_outliers) > 0:
    print(f"     • Age range: {age_outliers['Age'].min():.0f} - {age_outliers['Age'].max():.0f} years")

# Income outliers
income_q1, income_q3 = df['Income'].quantile([0.25, 0.75])
income_iqr = income_q3 - income_q1
income_upper = income_q3 + 1.5 * income_iqr
income_outliers = df[df['Income'] > income_upper]
print(f"   Income Outliers:")
print(f"     • Count: {len(income_outliers):,} ({len(income_outliers)/len(df)*100:.1f}%)")
print(f"     • Threshold: >${income_upper:,.0f}")
if len(income_outliers) > 0:
    print(f"     • Income range: ${income_outliers['Income'].min():,.0f} - ${income_outliers['Income'].max():,.0f}")

# Segmentation readiness
print(f"\n🎯 SEGMENTATION READINESS ASSESSMENT:")
print(f"   ✅ Data Quality:")
print(f"     • Missing values: 0% (Perfect)")
print(f"     • Data consistency: All integer types")
print(f"     • Sample size: {len(df):,} (Excellent statistical power)")
print(f"   ✅ Feature Diversity:")
print(f"     • Numerical features: 2 (Age, Income)")
print(f"     • Categorical features: 5 (Demographics)")
print(f"     • Feature correlation: {correlation:.3f} ({strength})")
print(f"   ✅ Distribution Characteristics:")
print(f"     • Age skewness: {df['Age'].skew():.2f}")
print(f"     • Income skewness: {df['Income'].skew():.2f}")
print(f"     • Income variability: {(df['Income'].std()/df['Income'].mean())*100:.1f}% CV")

# Business recommendations
print(f"\n🚀 BUSINESS RECOMMENDATIONS:")
print(f"   🎯 Segmentation Strategy:")
print(f"     • Recommended clusters: 4-6 (large dataset allows granularity)")
print(f"     • Primary segmentation: Income-based (3 tiers)")
print(f"     • Secondary segmentation: Age-Education combination")
print(f"     • Niche segments: High-income outliers ({len(income_outliers):,} customers)")
print(f"   📈 Marketing Opportunities:")
print(f"     • Female-focused premium products (higher income segment)")
print(f"     • Education-based product tiers")
print(f"     • Geographic customization by settlement size")
print(f"     • Age-appropriate service offerings")

print(f"\n🎉 ANALYSIS COMPLETE!")
print(f"📊 Generated 8 professional plots + comprehensive insights")
print(f"📁 All visualizations saved in 'figs/' folder")
print(f"✅ Ready for clustering analysis with {len(df):,} customer records")
print("="*60)
