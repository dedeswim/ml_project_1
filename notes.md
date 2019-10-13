# Notes about the project

TODO:

- [ ] understand features meaning
- [ ] decide features meaningfulness
- [ ] create a correlation matrix
- [ ] create first dataset without not meaningful and correlated features
- [ ] add unit tests (?)
- [ ] normalize the dataset

## 2019-10-10

There are a lot of datapoints with missing information in multiple features.
First thing to do would be to check all the features to check which ones have missing values [features_overview.ipynb](features_overview.ipynb).

Then, if missing values have a correlation across features (for all rows, the missing values always appear in the same columns)
we can proceed like this:

1) we decide what "non meaningful" means (e.g. % of missing values > 5%) (we should also check the description of features for
this, can be found in [documentation_v1.8.pdf](documentation_v1.8.pdf)
2) we create two datasets:
    - one with not meaningful features stripped away, and with only datapoints without missing values
    - one with [all/only non meaningful?] features, and with only datapoints without missing values
3) we train two models on the two datasets and see which one is better

For the beginning we should focus on the first (bigger) dataset, leaving the other for a later date (if we'll have time).
