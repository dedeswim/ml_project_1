# Notes about the project

TODO:

- [x] create branches for different ML models
- [x] add cross validation functions
- [x] implement working polynomial augmentation functions
- [x] understand features meaning (@edoardo 18/10)
- [x] decide features meaningfulness (@all 20/10)
- [x] create the correlation matrix (@davide 17/10)
- [x] create first dataset without not meaningful and correlated features (@all 20/10)
- [x] add unit tests (?)
- [x] try classification with linear regression (@mari 20/10)
- [x] add assertions around, to be sure to use the functions correctly
- [x] create logistic regression functions (@edoardo 20/10)
- [x] normalize the dataset
- [x] complete documentation:
  - [x] load_csv_data
  - [x] standardize
  - [x] predict_labels
  - [x] create_csv_submission

## 2019-10-10 [Davide]

There are a lot of datapoints with missing information in multiple features.
First thing to do would be to check all the features to check which ones have missing values [features_overview.ipynb](features_overview.ipynb).

Then, if missing values have a correlation across features (for all rows, the missing values always appear in the same columns)
we can proceed like this:

1. we decide what "non meaningful" means (e.g. % of missing values > 5%) (we should also check the description of features for
this, can be found in [documentation_v1.8.pdf](documentation_v1.8.pdf)
2. we create two datasets:
    - one with not meaningful features stripped away, and with only datapoints without missing values
    - one with [all/only non meaningful?] features, and with only datapoints without missing values
3. we train two models on the two datasets and see which one is better

For the beginning we should focus on the first (bigger) dataset, leaving the other for a later date (if we'll have time).

### 2019-10-20 [Davide]

Two possible methods of feature selection:

- Univariate feature selection (only works well for independent features)
- Recursive Feature Elimination

So I guess we should:

0. (Regularize dataset?)
1. fit Logistic Regression to the initial clean dataset
2. Either:
    1. remove linear dependent features
    2. Univariate feature selection

    Or

    - Recursive Feature Elimination

3. Eval accuracy improvement

### 2019-10-24 [Edoardo]

After some trials I could implement Newton method for logistic regression. The trick is using the regularizer. Unfortunately, it does not seem to converge as fast as excepted, since the gradient norm stays pretty high.

I also tried with SGD in logistic regressiojn, getting decent results (.675 accuracy), but i guess we can improve it. The setting was 100k iterations and a .001 gamma.
