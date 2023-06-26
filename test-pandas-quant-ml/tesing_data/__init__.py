from pathlib import Path

import pandas as pd

DF_AAPL = pd.read_csv(Path(__file__).parent.joinpath("aapl.csv"), parse_dates=True, index_col="Date")
