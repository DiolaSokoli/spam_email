# Spam Email Detection

A Python project that classifies messages as spam or legitimate using a custom
Gaussian radial basis function (RBF) neural network.

## Dataset

The project reads `spam_email/spam_email/spam.csv`, using the `v1` label and
`v2` message columns with Latin-1 encoding. The dataset source is not documented
in this repository.

## Approach

- Encode the message labels and convert message text to TF-IDF features.
- Remove English stop words and limit the vocabulary to 400 features.
- Split the data into 80% training and 20% test sets with random seed 42.
- Use K-Means to find 12 RBF centers, then fit and evaluate the custom Gaussian
  RBF classifier.
- Print accuracy and a confusion matrix; save classifier and class-distribution
  plots in `classifier/` and `distribution/`.

## Run

From the repository root, install the dependencies and run the classifier:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
cd spam_email/spam_email
python spam_rbf_classifier.py
```

The script expects `spam.csv` in its current working directory.

## Results

The script prints the test accuracy and confusion matrix and saves plots when it
runs. No fixed metric values are documented in this README.
