# Data

## Source
Telco Customer Churn dataset (IBM sample data), via Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## License
Copyright authors (see Kaggle page for terms)

## How to obtain
```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

## Shape
7,043 rows × 21 columns

## Target variable
`Churn` (Yes/No) — binary classification

## Known issues (to address in validation/EDA)
- `TotalCharges` is stored as string, not numeric — contains some blank values for customers with 0 tenure
- `customerID` is a unique identifier, not a feature — must be excluded from modeling