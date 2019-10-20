# Notes about the project

TODO:

- [ ] create branches for different ML models
- [ ] add cross validation functions
- [ ] implement working polynomial augmentation functions
- [ ] understand features meaning (@edoardo 18/10)
- [ ] decide features meaningfulness (@all 20/10)
- [ ] create a correlation matrix (@davide 17/10)
- [ ] create first dataset without not meaningful and correlated features (@all 20/10)
- [ ] add unit tests (?)
- [ ] try classification with linear regression (@mari 20/10)
- [ ] add assertions around, to be sure to use the functions correctly
- [x] create logistic regression functions (@edoardo 20/10)
- [x] normalize the dataset
- [x] complete documentation:
  - [x] load_csv_data
  - [x] standardize
  - [x] predict_labels
  - [x] create_csv_submission

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
