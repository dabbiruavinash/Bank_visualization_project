import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn')

# Set random seed for reproducibility
np.random.seed(42)

# ======================
# DATA GENERATION (Simulated Bank Data)
# ======================

def generate_bank_data(num_customers=1000):
    """Generate synthetic bank data for visualization"""
    
    # Customer demographics
    customer_ids = range(1000, 1000 + num_customers)
    genders = np.random.choice(['Male', 'Female'], size=num_customers, p=[0.48, 0.52])
    ages = np.random.normal(loc=45, scale=15, size=num_customers).astype(int)
    ages = np.clip(ages, 18, 90)
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], 
                               size=num_customers, 
                               p=[0.6, 0.3, 0.1])
    
    # Account information
    account_types = np.random.choice(['Checking', 'Savings', 'Both'], 
                                    size=num_customers, 
                                    p=[0.5, 0.3, 0.2])
    tenure = np.random.exponential(scale=5, size=num_customers).astype(int)
    tenure = np.clip(tenure, 0, 30)
    
    # Financial metrics
    checking_balance = np.round(np.random.gamma(shape=2, scale=1000, size=num_customers), 2)
    savings_balance = np.round(np.random.gamma(shape=3, scale=1500, size=num_customers), 2)
    credit_score = np.random.normal(loc=650, scale=100, size=num_customers).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Transaction activity
    monthly_transactions = np.random.poisson(lam=15, size=num_customers)
    monthly_fees = np.round(np.random.exponential(scale=5, size=num_customers), 2)
    
    # Loan information
    has_loan = np.random.choice([0, 1], size=num_customers, p=[0.7, 0.3])
    loan_amount = np.round(np.where(has_loan, 
                                  np.random.gamma(shape=2, scale=5000, size=num_customers), 
                                  0), 2)
    
    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Location': locations,
        'AccountType': account_types,
        'Tenure': tenure,
        'CheckingBalance': checking_balance,
        'SavingsBalance': savings_balance,
        'CreditScore': credit_score,
        'MonthlyTransactions': monthly_transactions,
        'MonthlyFees': monthly_fees,
        'HasLoan': has_loan,
        'LoanAmount': loan_amount
    })
    
    return data

# Generate and load the data
bank_data = generate_bank_data(1500)

# ======================
# DATA PREPARATION
# ======================

# Create age groups
bins = [18, 25, 35, 50, 65, 90]
labels = ['18-24', '25-34', '35-49', '50-64', '65+']
bank_data['AgeGroup'] = pd.cut(bank_data['Age'], bins=bins, labels=labels, right=False)

# Create balance categories
bank_data['TotalBalance'] = bank_data['CheckingBalance'] + bank_data['SavingsBalance']
balance_bins = [0, 1000, 5000, 10000, 25000, 100000, float('inf')]
balance_labels = ['<1k', '1k-5k', '5k-10k', '10k-25k', '25k-100k', '100k+']
bank_data['BalanceCategory'] = pd.cut(bank_data['TotalBalance'], 
                                     bins=balance_bins, 
                                     labels=balance_labels)

# Create credit score categories
score_bins = [300, 580, 670, 740, 800, 850]
score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
bank_data['CreditCategory'] = pd.cut(bank_data['CreditScore'], 
                                    bins=score_bins, 
                                    labels=score_labels)

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def create_demographic_dashboard(data):
    """Create a dashboard showing customer demographics"""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Bank Customer Demographics Dashboard', fontsize=16, y=1.02)
    gs = GridSpec(3, 3, figure=fig)
    
    # Age distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data['Age'], bins=20, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Age Distribution')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    
    # Gender distribution
    ax2 = fig.add_subplot(gs[0, 1])
    gender_counts = data['Gender'].value_counts()
    ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['lightcoral', 'lightskyblue'], startangle=90)
    ax2.set_title('Gender Distribution')
    
    # Location distribution
    ax3 = fig.add_subplot(gs[0, 2])
    location_counts = data['Location'].value_counts()
    sns.barplot(x=location_counts.index, y=location_counts.values, 
                palette='viridis', ax=ax3)
    ax3.set_title('Customer Locations')
    ax3.set_xlabel('Location')
    ax3.set_ylabel('Count')
    
    # Age group distribution
    ax4 = fig.add_subplot(gs[1, 0])
    age_group_counts = data['AgeGroup'].value_counts().sort_index()
    sns.barplot(x=age_group_counts.index, y=age_group_counts.values, 
                palette='coolwarm', ax=ax4)
    ax4.set_title('Age Group Distribution')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Count')
    
    # Age vs. Gender
    ax5 = fig.add_subplot(gs[1, 1:])
    sns.boxplot(x='Gender', y='Age', data=data, palette='pastel', ax=ax5)
    ax5.set_title('Age Distribution by Gender')
    ax5.set_xlabel('Gender')
    ax5.set_ylabel('Age')
    
    # Location by Gender
    ax6 = fig.add_subplot(gs[2, :])
    cross_tab = pd.crosstab(data['Location'], data['Gender'])
    cross_tab.plot(kind='bar', stacked=True, color=['lightcoral', 'lightskyblue'], ax=ax6)
    ax6.set_title('Location Distribution by Gender')
    ax6.set_xlabel('Location')
    ax6.set_ylabel('Count')
    ax6.legend(title='Gender')
    
    plt.tight_layout()
    return fig

def create_financial_dashboard(data):
    """Create a dashboard showing financial metrics"""
    
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle('Bank Financial Metrics Dashboard', fontsize=16, y=1.02)
    gs = GridSpec(3, 2, figure=fig)
    
    # Account type distribution
    ax1 = fig.add_subplot(gs[0, 0])
    account_counts = data['AccountType'].value_counts()
    ax1.pie(account_counts, labels=account_counts.index, autopct='%1.1f%%',
            colors=['gold', 'lightgreen', 'lightblue'], startangle=90)
    ax1.set_title('Account Type Distribution')
    
    # Balance distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data['TotalBalance'], bins=30, kde=True, ax=ax2, color='mediumseagreen')
    ax2.set_title('Total Balance Distribution')
    ax2.set_xlabel('Total Balance ($)')
    ax2.set_ylabel('Count')
    
    # Balance categories
    ax3 = fig.add_subplot(gs[1, 0])
    balance_counts = data['BalanceCategory'].value_counts().sort_index()
    sns.barplot(x=balance_counts.index, y=balance_counts.values, 
                palette='Blues_d', ax=ax3)
    ax3.set_title('Balance Categories')
    ax3.set_xlabel('Balance Range ($)')
    ax3.set_ylabel('Count')
    
    # Credit score distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.histplot(data['CreditScore'], bins=20, kde=True, ax=ax4, color='purple')
    ax4.set_title('Credit Score Distribution')
    ax4.set_xlabel('Credit Score')
    ax4.set_ylabel('Count')
    
    # Credit score categories
    ax5 = fig.add_subplot(gs[2, 0])
    credit_counts = data['CreditCategory'].value_counts().sort_index()
    sns.barplot(x=credit_counts.index, y=credit_counts.values, 
                palette='RdYlGn', ax=ax5)
    ax5.set_title('Credit Score Categories')
    ax5.set_xlabel('Credit Rating')
    ax5.set_ylabel('Count')
    
    # Loan distribution
    ax6 = fig.add_subplot(gs[2, 1])
    loan_data = data[data['HasLoan'] == 1]
    sns.histplot(loan_data['LoanAmount'], bins=20, kde=True, ax=ax6, color='orange')
    ax6.set_title('Loan Amount Distribution (Customers with Loans)')
    ax6.set_xlabel('Loan Amount ($)')
    ax6.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def create_behavioral_dashboard(data):
    """Create a dashboard showing customer behavior metrics"""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Customer Behavior Dashboard', fontsize=16, y=1.02)
    gs = GridSpec(2, 2, figure=fig)
    
    # Tenure distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data['Tenure'], bins=20, kde=True, ax=ax1, color='teal')
    ax1.set_title('Customer Tenure (Years with Bank)')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Count')
    
    # Monthly transactions
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data['MonthlyTransactions'], bins=20, kde=True, ax=ax2, color='royalblue')
    ax2.set_title('Monthly Transactions Distribution')
    ax2.set_xlabel('Transactions per Month')
    ax2.set_ylabel('Count')
    
    # Monthly fees
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(data['MonthlyFees'], bins=20, kde=True, ax=ax3, color='crimson')
    ax3.set_title('Monthly Fees Distribution')
    ax3.set_xlabel('Fees per Month ($)')
    ax3.set_ylabel('Count')
    
    # Age vs. Balance
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x='Age', y='TotalBalance', data=data, 
                    hue='AccountType', palette='viridis', alpha=0.6, ax=ax4)
    ax4.set_title('Age vs. Total Balance by Account Type')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Total Balance ($)')
    ax4.legend(title='Account Type')
    
    plt.tight_layout()
    return fig

def create_insights_dashboard(data):
    """Create a dashboard with key insights and correlations"""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Key Insights and Correlations', fontsize=16, y=1.02)
    gs = GridSpec(2, 2, figure=fig)
    
    # Balance by age group
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='AgeGroup', y='TotalBalance', data=data, 
                palette='coolwarm', ax=ax1)
    ax1.set_title('Total Balance by Age Group')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Total Balance ($)')
    
    # Credit score by account type
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x='AccountType', y='CreditScore', data=data, 
                palette='Set2', ax=ax2)
    ax2.set_title('Credit Score by Account Type')
    ax2.set_xlabel('Account Type')
    ax2.set_ylabel('Credit Score')
    
    # Tenure vs. balance
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x='Tenure', y='TotalBalance', data=data, 
                    hue='CreditCategory', palette='RdYlGn', alpha=0.7, ax=ax3)
    ax3.set_title('Tenure vs. Total Balance by Credit Rating')
    ax3.set_xlabel('Tenure (Years)')
    ax3.set_ylabel('Total Balance ($)')
    ax3.legend(title='Credit Rating')
    
    # Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    numeric_cols = data.select_dtypes(include=['int64', 'float64'])
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    return fig

# ======================
# GENERATE ALL DASHBOARDS
# ======================

def generate_all_dashboards(data):
    """Generate and save all dashboard visualizations"""
    
    print("Generating bank analytics dashboards...")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('bank_analytics_dashboards'):
        os.makedirs('bank_analytics_dashboards')
    
    # Generate and save dashboards
    demographic_fig = create_demographic_dashboard(data)
    demographic_fig.savefig('bank_analytics_dashboards/demographic_dashboard.png', 
                          bbox_inches='tight', dpi=300)
    plt.close(demographic_fig)
    
    financial_fig = create_financial_dashboard(data)
    financial_fig.savefig('bank_analytics_dashboards/financial_dashboard.png', 
                         bbox_inches='tight', dpi=300)
    plt.close(financial_fig)
    
    behavioral_fig = create_behavioral_dashboard(data)
    behavioral_fig.savefig('bank_analytics_dashboards/behavioral_dashboard.png', 
                          bbox_inches='tight', dpi=300)
    plt.close(behavioral_fig)
    
    insights_fig = create_insights_dashboard(data)
    insights_fig.savefig('bank_analytics_dashboards/insights_dashboard.png', 
                        bbox_inches='tight', dpi=300)
    plt.close(insights_fig)
    
    print("Dashboards saved to 'bank_analytics_dashboards' directory.")

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    # Generate and save all dashboards
    generate_all_dashboards(bank_data)
    
    # Show one dashboard for demonstration (optional)
    demo_fig = create_demographic_dashboard(bank_data)
    plt.show()
