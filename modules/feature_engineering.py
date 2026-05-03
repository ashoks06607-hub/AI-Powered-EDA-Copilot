import numpy as np
import pandas as pd


def suggest_features(df):
    suggestions = []

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # 🔹 1. LOG TRANSFORMATION
    
    for col in numeric_cols:
        skew = df[col].skew()

        if skew > 1:
            suggestions.append({
                "feature": f"{col}_log",
                "type": "Transformation",
                "reason": f"{col} is highly skewed",
                "code": f"df['{col}_log'] = np.log1p(df['{col}'])"
            })

    
    # 🔹 2. BINNING (VERY IMPORTANT)
    
    for col in numeric_cols:
        if df[col].nunique() > 10:
            suggestions.append({
                "feature": f"{col}_binned",
                "type": "Binning",
                "reason": f"{col} can be grouped into ranges",
                "code": f"df['{col}_binned'] = pd.cut(df['{col}'], bins=5)"
            })

    
    # 🔹 3. INTERACTION FEATURES
    
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]

        suggestions.append({
            "feature": f"{col1}_x_{col2}",
            "type": "Feature Creation",
            "reason": "Interaction between two numeric features",
            "code": f"df['{col1}_x_{col2}'] = df['{col1}'] * df['{col2}']"
        })

    
    # 🔹 4. ENCODING
    
    for col in cat_cols:
        if df[col].nunique() < 10:
            suggestions.append({
                "feature": f"{col}_encoded",
                "type": "Encoding",
                "reason": "Convert categorical to numeric",
                "code": f"pd.get_dummies(df['{col}'])"
            })

    return suggestions
