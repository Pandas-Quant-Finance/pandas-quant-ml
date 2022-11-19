
Add two new functions to pandas series:
* make_stationary
* discretize

## Analyze Data

Q-Q plot function for each variable

## Transform data
Use an invertible flow to transform data suitable for machine learning
```
df = src.ml.transform(
    Flow("Open", "High", "Low", "Close") >> PositionalBar() >> Flows(
        Flow("gap") >> LambertGaussianizer() >> Rescale(),
        Flow("body") >> LambertGaussianizer() >> Rescale(),
        Flow("shadow") >> LambertGaussianizer() >> Rescale(),
        Flow("position") >> LambertGaussianizer() >> Rescale(),
    ),
    Flow("Volume") >> PercentChange() >> LogNormalizer()
)
```

## Construct a training Loop

* Split data into train, test, validation using various Splitters
* provide a loop of _n_ epochs for mini-batches of size _m_

