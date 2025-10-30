"""
Regression Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Configuration
filepath = '../final_data/final_weekly.csv'
save_outputs = True  # Set to True to save tables and figures

# Load and Prepare Data
data = pd.read_csv(filepath)
data['date'] = pd.to_datetime(data['date'])

# Create gas price shock variable
data['gas_ma52'] = data['gas_regular_all_formulations'].rolling(window=52, min_periods=26).mean()
data['gas_price_shock'] = ((data['gas_regular_all_formulations'] - data['gas_ma52']) / data['gas_ma52']) * 100

# Variables
treatment_var = 'gas_price_shock'
basic_controls = ['unemployment_claims']
additional_controls = ['federal_funds_effective_rate']

#---------------------------------------------------------
# Helper Functions
def format_coef(coef, se, pval):
    """Format coefficient with stars"""
    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    return f"{coef:.4f}{stars}", f"({se:.4f})"

def run_regressions(data, outcome, treatment, controls_basic, controls_add):
    """Run three specifications and return models"""
    vars_needed = [outcome, treatment] + controls_basic + controls_add
    reg_data = data[vars_needed].dropna()
    y = reg_data[outcome]

    # (1) No controls
    X1 = add_constant(reg_data[treatment])
    model1 = OLS(y, X1).fit(cov_type='HC1')

    # (2) Basic controls
    X2 = add_constant(reg_data[[treatment] + controls_basic])
    model2 = OLS(y, X2).fit(cov_type='HC1')

    # (3) All controls
    X3 = add_constant(reg_data[[treatment] + controls_basic + controls_add])
    model3 = OLS(y, X3).fit(cov_type='HC1')

    return model1, model2, model3, reg_data

def create_table(models, treatment):
    """Create results table"""
    rows = []

    # Coefficient row
    coef_row = ['Gas Price Shock']
    se_row = ['(% deviation from MA)']
    for m in models:
        c, s = format_coef(m.params[treatment], m.bse[treatment], m.pvalues[treatment])
        coef_row.append(c)
        se_row.append(s)
    rows.extend([coef_row, se_row, [''] * 4])

    # Controls
    rows.extend([
        ['Basic Controls', 'No', 'Yes', 'Yes'],
        ['Additional Controls', 'No', 'No', 'Yes'],
        [''] * 4
    ])

    # Stats
    rows.extend([
        ['Observations'] + [f"{int(m.nobs):,}" for m in models],
        ['R²'] + [f"{m.rsquared:.4f}" for m in models]
    ])

    return pd.DataFrame(rows, columns=['', '(1)', '(2)', '(3)'])

#---------------------------------------------------------
# Run Analysis for Consumer Loans
print("="*70)
print("TABLE 1: CONSUMER LOANS")
print("="*70)

models_loans = run_regressions(data, 'consumer_loans_banks', treatment_var,
                                basic_controls, additional_controls)
table1 = create_table(models_loans[:3], treatment_var)
print("\n" + table1.to_string(index=False))

# Key stats
m3 = models_loans[2]
reg_data_loans = models_loans[3]
coef = m3.params[treatment_var]
se = m3.bse[treatment_var]
pval = m3.pvalues[treatment_var]
mean_y = reg_data_loans['consumer_loans_banks'].mean()
std_x = reg_data_loans[treatment_var].std()

print(f"\nKey Results:")
print(f"  Coefficient: {coef:.4f} (SE = {se:.4f}, p = {pval:.4f})")
print(f"  Effect: {abs(coef/mean_y*100):.2f}% of mean")
print(f"  1 SD effect: ${abs(coef*std_x):.2f}B or {abs(coef*std_x/mean_y*100):.2f}% of mean")

#---------------------------------------------------------
# Run Analysis for Bank Deposits
print("\n" + "="*70)
print("TABLE 2: BANK DEPOSITS")
print("="*70)

models_deposits = run_regressions(data, 'deposits_banks', treatment_var,
                                   basic_controls, additional_controls)
table2 = create_table(models_deposits[:3], treatment_var)
print("\n" + table2.to_string(index=False))

# Key stats
m3 = models_deposits[2]
reg_data_deposits = models_deposits[3]
coef = m3.params[treatment_var]
se = m3.bse[treatment_var]
pval = m3.pvalues[treatment_var]
mean_y = reg_data_deposits['deposits_banks'].mean()
std_x = reg_data_deposits[treatment_var].std()

print(f"\nKey Results:")
print(f"  Coefficient: {coef:.4f} (SE = {se:.4f}, p = {pval:.4f})")
print(f"  Effect: {abs(coef/mean_y*100):.2f}% of mean")
print(f"  1 SD effect: ${abs(coef*std_x):.2f}B or {abs(coef*std_x/mean_y*100):.2f}% of mean")

#---------------------------------------------------------
# Create Figure
print("\n" + "="*70)
print("CREATING FIGURE")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Consumer Loans
scatter_loans = data[['gas_price_shock', 'consumer_loans_banks']].dropna()
axes[0].scatter(scatter_loans['gas_price_shock'], scatter_loans['consumer_loans_banks'],
                alpha=0.3, s=20, color='steelblue', edgecolors='none')
z = np.polyfit(scatter_loans['gas_price_shock'], scatter_loans['consumer_loans_banks'], 1)
p = np.poly1d(z)
x_line = np.linspace(scatter_loans['gas_price_shock'].min(),
                     scatter_loans['gas_price_shock'].max(), 100)
axes[0].plot(x_line, p(x_line), "r-", linewidth=2.5, label='Fitted Line')
axes[0].set_xlabel('Gas Price Shock (% deviation from trend)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Consumer Loans ($ Billions)', fontsize=13, fontweight='bold')
axes[0].set_title('Gas Price Shocks and Consumer Borrowing', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Right: Bank Deposits
scatter_deposits = data[['gas_price_shock', 'deposits_banks']].dropna()
axes[1].scatter(scatter_deposits['gas_price_shock'], scatter_deposits['deposits_banks'],
                alpha=0.3, s=20, color='coral', edgecolors='none')
z = np.polyfit(scatter_deposits['gas_price_shock'], scatter_deposits['deposits_banks'], 1)
p = np.poly1d(z)
x_line = np.linspace(scatter_deposits['gas_price_shock'].min(),
                     scatter_deposits['gas_price_shock'].max(), 100)
axes[1].plot(x_line, p(x_line), "r-", linewidth=2.5, label='Fitted Line')
axes[1].set_xlabel('Gas Price Shock (% deviation from trend)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Bank Deposits ($ Billions)', fontsize=13, fontweight='bold')
axes[1].set_title('Gas Price Shocks and Bank Deposits', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

plt.tight_layout()

if save_outputs:
    plt.savefig('../final_data/figure1.png', dpi=300, bbox_inches='tight')
    table1.to_csv('../final_data/table1.csv', index=False)
    table2.to_csv('../final_data/table2.csv', index=False)
    print("\n✓ Saved: figure1.png, table1.csv, table2.csv")
else:
    plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)