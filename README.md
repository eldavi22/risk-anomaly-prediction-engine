# Risk Anomaly Prediction Engine

I built this project to explore how a nonlinear model can identify unusual patterns in risk data without relying on a machine learning framework. The classifier is written in Python using NumPy for the math and SciPy for optimization. It includes the forward pass, analytical gradients, training, and decision-threshold calibration.

The model uses one hidden layer with a tanh activation and a sigmoid output. Training minimizes regularized binary cross-entropy through SciPy's L-BFGS-B optimizer. After training, the script calibrates the classification threshold against validation predictions instead of assuming that 0.5 is always the best cutoff. It also evaluates the cost of missing higher-risk cases.

## Running the model

You'll need Python 3.10 or newer.

```bash
python -m pip install -r requirements-risk-anomaly.txt
python risk_anomaly_lbfgsb.py
```

The script generates its own synthetic dataset, trains the classifier, and prints the optimizer results, selected threshold, confusion matrix, precision, recall, and F1 score. You don't need to download a dataset.

The simulated data includes 6,000 examples with 10 input features. It combines several kinds of unusual activity, such as spikes in transaction amounts and frequency, inconsistent contextual signals, and nonlinear patterns that are harder to separate with a straight decision boundary.

## How I checked it

I compared the analytical gradients with finite differences to check the implementation. In a documented run using the default setup, the maximum gradient difference was approximately `2.67e-10`. Those results come from a local experiment with synthetic data, not a production benchmark.

The repository also has automated regression tests for gradients, training behavior, numerical stability, data splitting, risk penalties, and threshold calibration.

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the tests when changes are pushed or a pull request is opened.

## Project notes

This is a numerical machine learning project, not a hosted API. The main implementation is in `risk_anomaly_lbfgsb.py`, and the dependencies are listed in `requirements-risk-anomaly.txt`. I developed the original model separately and later added the test suite and CI configuration.

The source code is released under the MIT license. Third-party packages keep their own licenses.
