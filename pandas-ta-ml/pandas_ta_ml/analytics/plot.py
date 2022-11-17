import pandas as pd
import statsmodels.api as sm


def ml_qqplot(df: pd.DataFrame, figsize=8, fit=True, line='45', **kwargs):
    import matplotlib.pyplot as plt

    df = df.dropna()
    nr_columns = df.shape[1]
    nr_rows = (nr_columns + 1) // 2
    fig, ax = plt.subplots(nr_rows, 2, figsize=(figsize * 2, figsize * nr_rows))
    ax = ax.flatten()

    for i, col in enumerate(df.columns):
        ax[i].set_title(str(col))
        sm.qqplot(df[col], ax=ax[i], fit=fit, line=line, **kwargs)
