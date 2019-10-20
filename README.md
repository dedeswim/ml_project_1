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

Overleaf containing the LaTex report for the Project: https://www.overleaf.com/project/5d9efa0c65fb98000163917d

## Project structure

The project is structured in the following way:

```markdown
.
├── notes.md: general notes about the project development
├── README.md: this file :)
├── requirements.txt: contains the packages used to run the project
├── data: contains the datasets (.gitignore'd)
│   ├── test.csv
│   └── train.csv
├── notebooks
│   ├── features_overview.ipynb: notebook where we study the features
│   └── tests.ipynb: some random tests
└── src
    ├── main.py: will contain the final code to train the model
    ├── helpers.py
    ├── split.py
    ├── logistic: contains the functions used to train the logistic regression model
    │   ├── cost.py
    │   ├── gradient.py
    │   ├── hessian.py
    │   ├── implementations.py
    │   └── sigmoid.py
    └── regression: contains the functions used to train the linear regression model
        ├── gradient.py
        ├── implementations.py
        ├── loss.py
        ├── non_required_impl.py: some useful implementations that are not required by the project
        └── polynomials.py
```
