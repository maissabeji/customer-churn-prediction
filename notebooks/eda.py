import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    sns.boxplot(data=df, x='Churn', y=col, ax=ax)
    ax.set_title(f'{col} by Churn')
plt.tight_layout()
plt.savefig('notebooks/eda_numeric_vs_churn.png')
print("saved plot")

# Correlation check — quantify the tenure/TotalCharges relationship we hypothesized
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr())