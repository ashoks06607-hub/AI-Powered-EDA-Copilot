import pandas as pd


def generate_quality_report(df):
    report = []

    for col in df.columns:

        missing_pct = df[col].isnull().mean() * 100
        unique = df[col].nunique()

        # 🔴 Missing values
        if missing_pct > 30:
            report.append({
                "column": col,
                "issue": "High Missing Values",
                "detail": f"{missing_pct:.1f}% missing",
                "recommendation": "Consider dropping or imputing"
            })

        # 🔴 Constant column
        if unique == 1:
            report.append({
                "column": col,
                "issue": "Constant Column",
                "detail": "Only one unique value",
                "recommendation": "Drop this column"
            })

        # 🔴 High cardinality
        if unique > 100:
            report.append({
                "column": col,
                "issue": "High Cardinality",
                "detail": f"{unique} unique values",
                "recommendation": "Use encoding or grouping"
            })

        # 🔴 Skewness (only numeric)
        if df[col].dtype in ["int64", "float64"]:
            skew = df[col].skew()

            if abs(skew) > 1:
                report.append({
                    "column": col,
                    "issue": "High Skewness",
                    "detail": f"Skewness = {skew:.2f}",
                    "recommendation": "Apply log or transformation"
                })

    return pd.DataFrame(report)