import pandas as pd
import plotly.express as px


# =========================
# 📊 DATASET OVERVIEW
# =========================
def get_overview(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum()
    }


# =========================
# 📋 COLUMN SUMMARY
# =========================
def get_column_summary(df):
    summary = []

    for col in df.columns:
        summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": df[col].isnull().sum(),
            "unique": df[col].nunique()
        })

    return pd.DataFrame(summary)


# =========================
# 📈 NUMERIC DISTRIBUTIONS
# =========================
def plot_numeric_distributions(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    figs = []

    for col in num_cols:
        fig = px.histogram(df, x=col, title=f"{col} Distribution")
        figs.append(fig)

    return figs


# =========================
# 📊 CATEGORICAL DISTRIBUTIONS
# =========================
def plot_categorical_distributions(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    figs = []

    for col in cat_cols:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]

        fig = px.bar(vc, x=col, y="count", title=f"{col} Distribution")
        figs.append(fig)

    return figs


# =========================
# 🔗 BIVARIATE ANALYSIS
# =========================
def plot_bivariate(df):
    figs = []

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # numeric vs numeric
    for i in range(min(len(num_cols)-1, 3)):
        fig = px.scatter(
            df,
            x=num_cols[i],
            y=num_cols[i+1],
            title=f"{num_cols[i]} vs {num_cols[i+1]}"
        )
        figs.append(fig)

    # categorical vs numeric
    if len(cat_cols) > 0 and len(num_cols) > 0:
        for col in cat_cols[:2]:
            fig = px.box(
                df,
                x=col,
                y=num_cols[0],
                title=f"{col} vs {num_cols[0]}"
            )
            figs.append(fig)

    return figs


# =========================
# 🔥 CORRELATION HEATMAP
# =========================
def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    fig.update_layout(height=700)

    return fig