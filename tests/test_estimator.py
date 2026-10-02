"""The numpy IRLS logistic in twin/common.py must reproduce sklearn LogisticRegression(penalty='l2', C=1.0).
Run: python3 -m tests.test_estimator   (needs scikit-learn; not needed to run the twin itself)."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from twin.common import fit_l2_logistic, predict_proba

rng = np.random.default_rng(1)
worst_coef = worst_prob = 0.0
for trial in range(5):
    n, k = 20000, 8
    X = rng.normal(size=(n, k))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.1 + X @ rng.normal(scale=0.3, size=k))))).astype(float)
    w = fit_l2_logistic(X, y)
    m = LogisticRegression(penalty="l2", C=1.0, tol=1e-10, max_iter=5000).fit(X, y)
    worst_coef = max(worst_coef, np.abs(w[1:] - m.coef_[0]).max(), abs(w[0] - m.intercept_[0]))
    worst_prob = max(worst_prob, np.abs(predict_proba(w, X) - m.predict_proba(X)[:, 1]).max())
print(f"max |coef diff| {worst_coef:.2e}   max |prob diff| {worst_prob:.2e}")
assert worst_coef < 1e-4 and worst_prob < 1e-5
print("PASS")
