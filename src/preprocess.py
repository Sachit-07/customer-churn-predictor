import pandas as pd

def build_feature_vector(raw_input: dict, feature_columns: list, scaler, numerical_cols: list) -> pd.DataFrame:
    row = pd.DataFrame([0] * len(feature_columns), index=feature_columns).T
    row = row.astype(float)

    for col in numerical_cols:
        if col in raw_input:
            row.at[0, col] = raw_input[col]

    for field, value in raw_input.items():
        if field in numerical_cols:
            continue
        dummy_col = f"{field}_{value}"
        if dummy_col in row.columns:
            row.at[0, dummy_col] = 1

    row[numerical_cols] = scaler.transform(row[numerical_cols])

    return row[feature_columns]