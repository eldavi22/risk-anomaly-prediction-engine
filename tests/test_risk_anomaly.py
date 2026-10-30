import unittest
import numpy as np
from risk_anomaly_lbfgsb import (
    NonlinearRiskClassifier, TrainingConfig, stable_sigmoid,
    standardize_train_test, stratified_split, evaluate_predictions,
    calibrate_f1_threshold, generate_synthetic_risk_data,
)


class RiskAnomalyTests(unittest.TestCase):
    def test_sigmoid_remains_finite_for_extremes(self):
        values = stable_sigmoid(np.array([-1000., 0., 1000.]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.allclose(values, [0., .5, 1.]))

    def test_standardization_fits_train_only(self):
        train = np.array([[1., 2.], [3., 4.], [5., 6.]])
        transformed_train, other = standardize_train_test(train, np.array([[1000., -1000.]]))
        self.assertTrue(np.allclose(transformed_train.mean(axis=0), 0.))
        self.assertTrue(np.allclose(transformed_train.std(axis=0), 1.))
        self.assertTrue(other[0, 0] > 100.)

    def test_stratified_split_retains_samples_and_labels(self):
        x = np.arange(100).reshape(50, 2).astype(float)
        y = np.array([0] * 15 + [1] * 10)
        splits = stratified_split(x, y, np.random.default_rng(4))
        self.assertEqual(sum(len(s.y) for s in splits), 25)
        self.assertEqual(sum(np.sum(s.y) for s in splits), 10)
        self.assertTrue(all(set(np.unique(s.y)) == {0, 1} for s in splits))

    def test_invalid_split_proportions_fail(self):
        x = np.zeros((20, 2))
        with self.assertRaises(ValueError):
            stratified_split(x, np.zeros(20, dtype=int), np.random.default_rng(4), .9, .2)

    def test_confusion_and_asymmetric_penalty(self):
        result = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([.2, .9, .1, .8]), .5)
        self.assertEqual((result.true_negative, result.false_positive, result.false_negative, result.true_positive), (1, 1, 1, 1))
        self.assertAlmostEqual(result.f1, .5)
        self.assertAlmostEqual(result.false_negative_risk_penalty, 2.)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([1]), np.array([.9]), 1.1)

    def test_invalid_model_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            NonlinearRiskClassifier(n_features=0, hidden_units=2, l2_strength=.01, seed=1)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(9, 3))
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])
        model = NonlinearRiskClassifier(n_features=3, hidden_units=3, l2_strength=.02, seed=8)
        p = model.parameters.copy()
        _, analytic = model.loss_and_gradient(p, x, y)
        delta = 1e-6
        for k in range(len(p)):
            pos, neg = p.copy(), p.copy()
            pos[k] += delta
            neg[k] -= delta
            fd = (model.loss_and_gradient(pos, x, y)[0] - model.loss_and_gradient(neg, x, y)[0]) / (2 * delta)
            self.assertAlmostEqual(fd, analytic[k], delta=1e-5)

    def test_probability_shape_and_bounds(self):
        model = NonlinearRiskClassifier(n_features=3, hidden_units=5, l2_strength=.01, seed=7)
        out = model.predict_proba(np.zeros((5, 3)))
        self.assertEqual(out.shape, (5,))
        self.assertTrue(np.all((out >= 0) & (out <= 1)))

    def test_f1_calibration_matches_discrete_optimum(self):
        y = np.array([1, 1, 0, 0, 1])
        probabilities = np.array([.9, .5, .7, .1, .4])
        threshold, _ = calibrate_f1_threshold(y, probabilities)
        best = max(evaluate_predictions(y, probabilities, float(t)).f1 for t in np.r_[0., probabilities, 1.])
        self.assertAlmostEqual(evaluate_predictions(y, probabilities, threshold).f1, best)

    def test_config_minimum_sample_size(self):
        with self.assertRaises(ValueError):
            generate_synthetic_risk_data(TrainingConfig(n_samples=10))

    def test_training_reduces_objective(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(25, 2))
        y = (x[:, 0] - x[:, 1] > 0).astype(int)
        model = NonlinearRiskClassifier(n_features=2, hidden_units=3, l2_strength=.01, seed=1)
        before, _ = model.loss_and_gradient(model.parameters, x, y)
        model.fit(x, y, max_iterations=35)
        after, _ = model.loss_and_gradient(model.parameters, x, y)
        self.assertLess(after, before)

if __name__ == '__main__':
    unittest.main()
