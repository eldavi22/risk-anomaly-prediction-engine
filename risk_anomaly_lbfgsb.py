"""Motor predictivo de anomalías de riesgo multidimensional.

Dependencias permitidas: numpy y scipy. El modelo es una red neuronal de una
capa oculta implementada desde cero y entrenada con L-BFGS-B usando gradientes
analíticos de la entropía cruzada regularizada.

Ejecución:
    python risk_anomaly_lbfgsb.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.stats import beta, gamma, lognorm, t


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class DatasetSplit(NamedTuple):
    """Partición de datos con etiquetas binarias."""

    x: FloatArray
    y: IntArray


@dataclass(frozen=True)
class Metrics:
    """Métricas binarias calculadas sin bibliotecas de ML."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    precision: float
    recall: float
    f1: float
    false_negative_risk_penalty: float


@dataclass(frozen=True)
class TrainingConfig:
    n_samples: int = 6_000
    n_features: int = 10
    hidden_units: int = 24
    l2_strength: float = 2e-3
    max_iterations: int = 250
    random_seed: int = 20261026


def stable_sigmoid(values: FloatArray) -> FloatArray:
    """Sigmoide numéricamente estable para evitar overflow en exp()."""
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def standardize_train_test(x_train: FloatArray, x_other: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Estandariza usando exclusivamente estadísticas del conjunto de entrenamiento."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x_train - mean) / scale, (x_other - mean) / scale


def stratified_split(
    x: FloatArray, y: IntArray, rng: np.random.Generator, train_ratio: float = 0.65, validation_ratio: float = 0.20
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Partición reproducible que preserva la proporción de anomalías."""
    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("Las proporciones de train/validation deben sumar menos de 1.")

    indices_by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    splits: list[list[IntArray]] = [[], [], []]
    for indices in indices_by_class:
        shuffled = rng.permutation(indices)
        first = int(len(shuffled) * train_ratio)
        second = first + int(len(shuffled) * validation_ratio)
        for bucket, selection in zip(splits, (shuffled[:first], shuffled[first:second], shuffled[second:])):
            bucket.append(selection)

    result = []
    for bucket in splits:
        indices = rng.permutation(np.concatenate(bucket))
        result.append(DatasetSplit(x=x[indices], y=y[indices]))
    return result[0], result[1], result[2]


def generate_synthetic_risk_data(config: TrainingConfig) -> tuple[FloatArray, IntArray]:
    """Genera transacciones normales y anomalías con distribuciones no gaussianas.

    Las anomalías se crean en tres regímenes: picos de importe/velocidad,
    inconsistencias de identidad-operación y patrones no lineales en un anillo.
    Las distribuciones lognormal, t de Student, beta y gamma modelan colas,
    asimetrías y concentraciones que una gaussiana no captura bien.
    """
    if config.n_samples < 5_000 or config.n_features < 10:
        raise ValueError("El proyecto requiere al menos 5,000 muestras y 10 características.")

    rng = np.random.default_rng(config.random_seed)
    anomaly_rate = 0.13
    y = (rng.random(config.n_samples) < anomaly_rate).astype(np.int_)
    x = np.empty((config.n_samples, config.n_features), dtype=np.float64)

    # Base normal: características heterogéneas y no gaussianas.
    x[:, 0] = lognorm(s=0.48, scale=np.exp(3.6)).rvs(size=config.n_samples, random_state=rng)  # importe
    x[:, 1] = gamma(a=2.3, scale=1.25).rvs(size=config.n_samples, random_state=rng)  # velocidad
    x[:, 2] = beta(a=4.2, b=2.8).rvs(size=config.n_samples, random_state=rng)  # confianza de dispositivo
    x[:, 3] = t(df=7).rvs(size=config.n_samples, random_state=rng)  # desviación geográfica
    x[:, 4] = beta(a=2.2, b=6.5).rvs(size=config.n_samples, random_state=rng)  # ratio nocturno
    x[:, 5] = rng.uniform(-1.0, 1.0, size=config.n_samples)  # patrón temporal codificado
    x[:, 6] = gamma(a=3.0, scale=0.8).rvs(size=config.n_samples, random_state=rng)  # antigüedad/actividad
    x[:, 7] = t(df=5).rvs(size=config.n_samples, random_state=rng) * 0.65  # ruido de red
    x[:, 8] = beta(a=3.5, b=4.0).rvs(size=config.n_samples, random_state=rng)  # consistencia documental
    x[:, 9] = lognorm(s=0.35, scale=np.exp(1.9)).rvs(size=config.n_samples, random_state=rng)  # exposición

    anomalous = np.flatnonzero(y == 1)
    modes = rng.integers(0, 3, size=len(anomalous))

    # Modo 0: importe y ráfaga inusuales, con colas muy pesadas.
    idx = anomalous[modes == 0]
    x[idx, 0] *= lognorm(s=0.70, scale=np.exp(1.15)).rvs(size=len(idx), random_state=rng)
    x[idx, 1] += gamma(a=3.5, scale=2.1).rvs(size=len(idx), random_state=rng)
    x[idx, 7] += t(df=2.4).rvs(size=len(idx), random_state=rng) * 2.6
    x[idx, 9] *= lognorm(s=0.55, scale=np.exp(0.65)).rvs(size=len(idx), random_state=rng)

    # Modo 1: baja consistencia y combinación geográfica/horaria anómala.
    idx = anomalous[modes == 1]
    x[idx, 2] = beta(a=1.0, b=7.0).rvs(size=len(idx), random_state=rng)
    x[idx, 4] = beta(a=7.5, b=1.4).rvs(size=len(idx), random_state=rng)
    x[idx, 3] += t(df=2.8).rvs(size=len(idx), random_state=rng) * 3.0
    x[idx, 8] = beta(a=1.1, b=6.2).rvs(size=len(idx), random_state=rng)

    # Modo 2: región anular no lineal, difícil de separar con una sola recta.
    idx = anomalous[modes == 2]
    angle = rng.uniform(0.0, 2.0 * np.pi, size=len(idx))
    radius = rng.normal(loc=1.55, scale=0.18, size=len(idx))
    x[idx, 5] = radius * np.cos(angle)
    x[idx, 3] = radius * np.sin(angle) + t(df=9).rvs(size=len(idx), random_state=rng) * 0.10
    x[idx, 6] += gamma(a=2.0, scale=1.2).rvs(size=len(idx), random_state=rng)
    x[idx, 0] *= lognorm(s=0.35, scale=np.exp(0.30)).rvs(size=len(idx), random_state=rng)

    # Ruido estocástico compartido y dependencia débil entre señales.
    x[:, 1] += 0.20 * np.log1p(x[:, 0]) + 0.15 * x[:, 4] * x[:, 9]
    x[:, 6] += 0.18 * np.abs(x[:, 7])
    return x, y


class NonlinearRiskClassifier:
    """Clasificador MLP (entrada -> tanh -> sigmoide) implementado con NumPy.

    Para z1 = XW1 + b1, h = tanh(z1), z2 = hW2 + b2 y p = sigmoid(z2),
    la derivada de la entropía cruzada logística respecto de z2 es (p - y)/n.
    Por regla de la cadena, dL/dz1 = (dL/dz2 W2^T) * (1 - h^2). Eso permite
    devolver el jacobiano exacto a L-BFGS-B sin diferenciación numérica.
    """

    def __init__(self, n_features: int, hidden_units: int, l2_strength: float, seed: int) -> None:
        if n_features <= 0 or hidden_units <= 0 or l2_strength < 0:
            raise ValueError("Dimensiones y regularización inválidas.")
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.l2_strength = l2_strength
        self._rng = np.random.default_rng(seed)
        self.parameters = self._initial_parameters()
        self.result: OptimizeResult | None = None

    @property
    def parameter_count(self) -> int:
        return self.n_features * self.hidden_units + self.hidden_units + self.hidden_units + 1

    def _initial_parameters(self) -> FloatArray:
        # Xavier/Glorot: mantiene la varianza aproximadamente estable entre capas.
        w1_limit = np.sqrt(6.0 / (self.n_features + self.hidden_units))
        w2_limit = np.sqrt(6.0 / (self.hidden_units + 1))
        w1 = self._rng.uniform(-w1_limit, w1_limit, size=(self.n_features, self.hidden_units))
        b1 = np.zeros(self.hidden_units, dtype=np.float64)
        w2 = self._rng.uniform(-w2_limit, w2_limit, size=(self.hidden_units, 1))
        b2 = np.zeros(1, dtype=np.float64)
        return self._pack(w1, b1, w2, b2)

    def _pack(self, w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray) -> FloatArray:
        return np.concatenate((w1.ravel(), b1, w2.ravel(), b2)).astype(np.float64, copy=False)

    def _unpack(self, parameters: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        if parameters.size != self.parameter_count:
            raise ValueError("Vector de parámetros con dimensión incorrecta.")
        split_w1 = self.n_features * self.hidden_units
        split_b1 = split_w1 + self.hidden_units
        split_w2 = split_b1 + self.hidden_units
        w1 = parameters[:split_w1].reshape(self.n_features, self.hidden_units)
        b1 = parameters[split_w1:split_b1]
        w2 = parameters[split_b1:split_w2].reshape(self.hidden_units, 1)
        b2 = parameters[split_w2:]
        return w1, b1, w2, b2

    def _forward(self, x: FloatArray, parameters: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        w1, b1, w2, b2 = self._unpack(parameters)
        hidden = np.tanh(x @ w1 + b1)  # broadcasting de b1 por muestra
        logits = (hidden @ w2 + b2).ravel()
        return hidden, logits, stable_sigmoid(logits)

    def loss_and_gradient(self, parameters: FloatArray, x: FloatArray, y: IntArray) -> tuple[float, FloatArray]:
        """Entropía cruzada estable + L2 y gradiente analítico de todas las capas."""
        if x.ndim != 2 or x.shape[1] != self.n_features or y.shape[0] != x.shape[0]:
            raise ValueError("Dimensiones incompatibles entre X, y y el modelo.")
        w1, _b1, w2, _b2 = self._unpack(parameters)
        hidden, logits, probabilities = self._forward(x, parameters)
        y_float = y.astype(np.float64)
        # logaddexp(0, z) - y*z equivale a BCE y es estable para logits extremos.
        data_loss = np.mean(np.logaddexp(0.0, logits) - y_float * logits)
        regularization = 0.5 * self.l2_strength * (np.sum(w1 * w1) + np.sum(w2 * w2))

        n_samples = x.shape[0]
        delta_output = ((probabilities - y_float) / n_samples).reshape(-1, 1)
        grad_w2 = hidden.T @ delta_output + self.l2_strength * w2
        grad_b2 = delta_output.sum(axis=0)
        delta_hidden = (delta_output @ w2.T) * (1.0 - hidden * hidden)
        grad_w1 = x.T @ delta_hidden + self.l2_strength * w1
        grad_b1 = delta_hidden.sum(axis=0)
        return float(data_loss + regularization), self._pack(grad_w1, grad_b1, grad_w2, grad_b2)

    def fit(self, x_train: FloatArray, y_train: IntArray, max_iterations: int) -> OptimizeResult:
        """Optimiza pesos con L-BFGS-B y gradientes exactos.

        L-BFGS-B aproxima el Hessiano inverso usando un historial limitado de
        gradientes. Aunque no se imponen cotas numéricas a los pesos, el método
        conserva su esquema cuasi-Newton y suele converger con muchas menos
        iteraciones que descenso de gradiente de paso fijo.
        """

        objective: Callable[[FloatArray], tuple[float, FloatArray]] = lambda p: self.loss_and_gradient(p, x_train, y_train)
        try:
            result = minimize(
                fun=objective,
                x0=self.parameters,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iterations, "ftol": 1e-7, "gtol": 2e-4, "maxls": 30},
            )
        except (FloatingPointError, ValueError) as exc:
            raise RuntimeError(f"Falló la optimización L-BFGS-B: {exc}") from exc
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("L-BFGS-B devolvió parámetros no finitos.")
        self.parameters = result.x.astype(np.float64, copy=True)
        self.result = result
        return result

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """Probabilidad estimada de anomalía de riesgo."""
        return self._forward(x, self.parameters)[2]


def evaluate_predictions(y_true: IntArray, probabilities: FloatArray, threshold: float) -> Metrics:
    """Construye la matriz de confusión y métricas con operaciones NumPy."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    predicted = (probabilities >= threshold).astype(np.int_)
    true_negative = int(np.sum((y_true == 0) & (predicted == 0)))
    false_positive = int(np.sum((y_true == 0) & (predicted == 1)))
    false_negative = int(np.sum((y_true == 1) & (predicted == 0)))
    true_positive = int(np.sum((y_true == 1) & (predicted == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    # Coste asimétrico: un FN de riesgo cuesta 7 veces una alerta falsa.
    risk_penalty = (7.0 * false_negative + false_positive) / max(len(y_true), 1)
    return Metrics(true_negative, false_positive, false_negative, true_positive, precision, recall, f1, risk_penalty)


def calibrate_f1_threshold(y_validation: IntArray, probabilities: FloatArray) -> tuple[float, OptimizeResult]:
    """Calibra un umbral con SciPy y garantiza el máximo F1 entre cortes reales.

    F1 es una función escalonada de la probabilidad. Se usa minimize_scalar
    sobre [0, 1] como búsqueda continua solicitada y luego se contrasta el
    candidato con todos los cortes inducidos por probabilidades observadas.
    Así el resultado final es exacto sobre las decisiones que pueden cambiar.
    """

    def negative_f1(threshold: float) -> float:
        return -evaluate_predictions(y_validation, probabilities, float(threshold)).f1

    scalar_result = minimize_scalar(
        negative_f1,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    candidate_thresholds = np.unique(
        np.concatenate((np.array([0.0, 1.0, scalar_result.x]), probabilities))
    )
    f1_values = np.array(
        [evaluate_predictions(y_validation, probabilities, float(value)).f1 for value in candidate_thresholds]
    )
    best_index = int(np.argmax(f1_values))
    return float(candidate_thresholds[best_index]), scalar_result


def print_metrics(title: str, metrics: Metrics, threshold: float) -> None:
    """Presentación por consola de métricas solicitadas."""
    print(f"\n{title}")
    print(f"Threshold F1: {threshold:.6f}")
    print("Matriz de confusión [filas=real, columnas=predicción]")
    print(f"[[TN={metrics.true_negative:4d}, FP={metrics.false_positive:4d}],")
    print(f" [FN={metrics.false_negative:4d}, TP={metrics.true_positive:4d}]]")
    print(f"Precisión: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1-Score:  {metrics.f1:.4f}")
    print(f"Penalización de Falso Negativo de Riesgo: {metrics.false_negative_risk_penalty:.4f}")


def main() -> None:
    """Orquesta datos, entrenamiento, calibración y evaluación final."""
    config = TrainingConfig()
    try:
        x, y = generate_synthetic_risk_data(config)
        rng = np.random.default_rng(config.random_seed + 1)
        train_raw, validation_raw, test_raw = stratified_split(x, y, rng)
        x_train, x_validation = standardize_train_test(train_raw.x, validation_raw.x)
        # Reutilizamos media/desvío de train para test sin filtrar información.
        train_mean, train_std = train_raw.x.mean(axis=0), train_raw.x.std(axis=0)
        train_std = np.where(train_std < 1e-12, 1.0, train_std)
        x_test = (test_raw.x - train_mean) / train_std

        model = NonlinearRiskClassifier(
            n_features=config.n_features,
            hidden_units=config.hidden_units,
            l2_strength=config.l2_strength,
            seed=config.random_seed,
        )
        result = model.fit(x_train, train_raw.y, config.max_iterations)
        validation_probabilities = model.predict_proba(x_validation)
        threshold, scalar_result = calibrate_f1_threshold(validation_raw.y, validation_probabilities)
        test_probabilities = model.predict_proba(x_test)
        test_metrics = evaluate_predictions(test_raw.y, test_probabilities, threshold)

        print("Motor Predictivo de Anomalías de Riesgo Multidimensional")
        print(f"Muestras: {len(y)} | Características: {config.n_features} | Parámetros: {model.parameter_count}")
        print(f"Anomalías sintéticas: {y.mean():.2%}")
        print(f"L-BFGS-B: success={result.success} | iteraciones={result.nit} | pérdida final={result.fun:.6f}")
        print(f"minimize_scalar: success={scalar_result.success} | umbral continuo={scalar_result.x:.6f}")
        print_metrics("Evaluación de prueba (test independiente)", test_metrics, threshold)
    except (RuntimeError, ValueError) as exc:
        print(f"Error controlado: {exc}")


if __name__ == "__main__":
    main()
