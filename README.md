# Motor Predictivo de Anomalías de Riesgo Multidimensional

Clasificador no lineal de anomalías construido desde cero con **NumPy** y
**SciPy**. No utiliza scikit-learn, pandas, TensorFlow, Keras ni PyTorch.

## Contenido

- `risk_anomaly_lbfgsb.py`: script único, orientado a objetos y ejecutable.
- `requirements-risk-anomaly.txt`: dependencias permitidas.

## Requisitos

- Python 3.10 o superior.
- NumPy.
- SciPy.

Instalación aislada recomendada en Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-risk-anomaly.txt
```

## Ejecución

```powershell
.venv\Scripts\python.exe risk_anomaly_lbfgsb.py
```

El programa crea un dataset sintético y muestra por consola:

- estado, iteraciones y pérdida final de L-BFGS-B;
- umbral de decisión calibrado con F1;
- matriz de confusión;
- precisión, recall y F1-score;
- penalización asimétrica de falsos negativos de riesgo.

## Diseño matemático

El modelo es una red de una capa oculta:

```text
X -> (W1, b1) -> tanh -> (W2, b2) -> sigmoid -> P(anomalía)
```

La función objetivo es entropía cruzada binaria regularizada con L2. El script
calcula el gradiente analítico por regla de la cadena y lo entrega a
`scipy.optimize.minimize` con método `L-BFGS-B`.

Para la calibración, `scipy.optimize.minimize_scalar` explora el intervalo de
umbrales `[0, 1]`. Como F1 es escalonada, el resultado se contrasta además con
los cortes inducidos por las probabilidades de validación: así se conserva el
uso de SciPy y se selecciona el mejor umbral que puede cambiar decisiones.

## Datos sintéticos

Se generan 6,000 transacciones con 10 características heterogéneas. Las
distribuciones `lognorm`, `gamma`, `beta` y t de Student inyectan asimetría,
colas pesadas y ruido no gaussiano. Las anomalías combinan tres regímenes:

1. picos de importe, velocidad y exposición;
2. inconsistencias de dispositivo, horario y geografía;
3. una región anular no lineal que no es separable con una única frontera lineal.

## Validación realizada

Con la semilla por defecto se verificó que L-BFGS-B converge y que el gradiente
analítico coincide con diferencias finitas, con error absoluto máximo aproximado
de `2.67e-10` sobre una muestra de parámetros.

## Restricciones de dependencias

El script sólo importa NumPy, SciPy y módulos de la biblioteca estándar de
Python (`dataclasses` y `typing`).

## Automated regression tests

Run `python -m unittest discover -s tests -v`. Twelve tests cover numerical stability, gradients, training, risk penalty, stratified splitting and threshold calibration. GitHub Actions reruns them on each push or pull request.

## Scope and provenance

This is a Python **machine-learning model**, not an HTTP API. It generates synthetic data and implements a nonlinear classifier using NumPy and SciPy. The performance metrics described are from recorded local runs, not guaranteed on unrelated datasets or production infrastructure.

The released source is based on the owner's archived `risk_anomaly_lbfgsb_package.zip`, with original model code preserved and new tests/CI added separately. License terms apply to repository source only, not external datasets or third-party libraries.
