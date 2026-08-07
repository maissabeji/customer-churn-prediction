import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ['Contract', 'InternetService', 'PaymentMethod']):
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())
    churn_rate.plot(kind='bar', ax=ax)
    ax.set_title(f'Churn rate by {col}')
    ax.set_ylabel('Churn rate')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('notebooks/eda_categorical_vs_churn.png')
print("saved plot")