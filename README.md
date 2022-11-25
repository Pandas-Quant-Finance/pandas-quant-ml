
Core Functionality of this module:
* make data stationary
* discretize
* analyze data distribution (q-q plot)
* measure quality of fit


## Transform data
Use an invertible flow to transform data suitable for machine learning
```
transformed_df, inverter = df.ml.transform(
    SelectJoin(
        Select("Open", "High", "Low", "Close") >> GapUpperLowerBody() >> SelectJoin(
            Select("gap") >> LambertGaussianizer() >> Rescale(),
            Select("upper") >> LambertGaussianizer() >> Rescale(),
            Select("lower") >> LambertGaussianizer() >> Rescale(),
            Select("body") >> LambertGaussianizer() >> Rescale(),
        ),
        Select("Volume") >> PercentChange() >> LogNormalizer() >> Rescale()
    ),
    return_inverter=True
)

original = inv(transformed_df)
```

To transform data for machine learning a similar method can be used to extract / transform
features and labels out of a single source DataFrame.

```
features, labels, label_inverter = df.ml.features_labels(
    # features        
    SelectJoin(
        Select("Open", "High", "Low", "Close") >> GapUpperLowerBody(),
        Select("Volume") >> PercentChange() >> LogNormalizer()
    ),
    # labels
    Select("Close") >> PercentChange(),
    # predict 5 steps into the future
    labels_shift=-5
)
```

## Analyze Data

Q-Q plot function for each variable:
```
df.ml.transform(
    SelectJoin(
        Select("Close", rename='LogReturns') >> PercentChange() >> LogNormalizer(),
        Select("Close", rename='Lambert') >> PercentChange() >> LambertGaussianizer(),
    )
).ml.qqplot()
```


## Construct a training Loop

* Split data into train, test, validation using various Splitters
* provide a loop of _n_ epochs for mini-batches of size _m_

