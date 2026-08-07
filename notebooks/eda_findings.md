# EDA Findings — Customer Churn

## Numeric features
- **tenure**: strongest apparent signal. Non-churners median ~38 months, churners median ~10 months.
- **MonthlyCharges**: churners pay more (median ~$80 vs ~$64).
- **TotalCharges**: correlated with tenure (r=0.83) — near-redundant, likely arithmetic byproduct (tenure × MonthlyCharges). Decision: drop in feature engineering.

## Categorical features
- **Contract**: strongest categorical signal. Month-to-month churn ~43%, One year ~11%, Two year ~3% (~14x gap). Business-actionable — contract upgrades are a viable retention lever.
- **InternetService**: Fiber optic churns ~42% vs DSL ~19%. Partly explained by price — fiber optic median MonthlyCharges ($91.68) far exceeds DSL ($56.15). Keep both features, monitor overlap via feature importance.
- **PaymentMethod**: Electronic check churns ~45% vs ~15-19% for other methods. Entangled with Contract — 47.7% of month-to-month customers use electronic check vs 9.9% of two-year customers. Keep both, monitor overlap.

## Decisions carried into Feature Engineering (Milestone 4)
1. Drop `TotalCharges` (redundant with tenure)
2. Normalize `SeniorCitizen` (0/1) to match Yes/No convention of other binary columns
3. Handle "No internet service" / "No phone service" categories — structural, not missing data
4. Keep InternetService + MonthlyCharges, Contract + PaymentMethod despite overlap — resolve via feature importance, not manual dropping