# ML Project 1

## General info

Repository containing the code for the [Project 1](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf) of the Machine Learning [course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at EPFL.

The team (SchroedingerCats) is composed by:

- Edoardo Debenedetti ([@dedeswim](https://github.com/dedeswim))
- Mari Sofie Lerfaldet ([@marisofie](https://github.com/marisofie))
- Davide Nanni ([@DSAureli](https://github.com/DSAureli))

The packages used to get the project up and running are listed in requirements.txt, and can be installed with:

```shell
pip install --user --requirement requirements.txt
```
For visualization purposes in the feature selection and engineering phase, we also used matplotlib, seaborn and pandas, but they are not needed to run the models and the final training.

## Project structure

The project is structured in the following way:

```markdown
.
├── implementations.py: contains **all the implementations** required by the project
├── notes.md: general notes about the project development
├── README.md: this file :)
├── requirements.txt: contains the packages used to run the project
├── run.py: contains the **final code** to train the model
├── tests.ipynb:a notebook that contains the tests of the required implementations
├── data: contains the datasets (.gitignore'd)
├── notebooks
│   ├── features_log.ipynb: contains our investigations about taking the logarithm of the features
│   ├── features_overview.ipynb: contains the exploratory data analysis phase
│   ├── logistic_regression.ipynb: contains out trials with logistic regression
│   └── ridge_regression.ipynb: contains our trials with ridge regression
└── src
    ├── helpers.py: some helper functions used by different modules
    ├── split.py: contains the function used to split the dataset into training and test sets
    ├── k_fold.py: contains the functions used for cross-validation
    ├── polynomials.py: contains the functions used to get the polynom
    ├── logistic: contains the functions used to train the logistic regression model
    │   ├── loss.py: contains the function to compute the loss function
    │   ├── gradient.py: contains the function to compute the gradient
    │   ├── hessian.py: contains the function to compute the hessian
    │   ├── implementations.py: contains the **logistic regression** implementations required by the project
    │   └── sigmoid.py: contains the function to compute the sigmoid
    └── linear: contains the functions used to train the linear regression model
        ├── gradient.py: contains the function to compute the gradient
        ├── implementations.py: contains the **linear regression** implementations required by the project
        └── loss.py: contains the function to compute the loss function
```
