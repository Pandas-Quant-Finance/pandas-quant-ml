
Add two new functions to pandas series:
* make_stationary
* discretize

Add Q-Q plot function for each variable

df.ml.transform(
    Col("Open", "High") >> lambda s: s.ml.zscore,
    ["Close"] >> lambda s: s.ml.zscore,
)

df.ml.transform(
    ["Open"], lambda s: s.ml.zscore
)(
    ["Close"], lambda s: s.ml.zscore,
)
    