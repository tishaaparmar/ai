import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
X_train = np.random.rand(200, 3)          # training data (used for mean / std)
x = np.array([0.9, 0.2, 0.7])             # instance to explain
feature_names = ["f0", "f1", "f2"]


def model(X):
    """toy classifier -> P(class 1)"""
    z = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] - 1
    return 1 / (1 + np.exp(-z))


# ================= IMPLEMENT =================
def lime_tabular(x, X_train, model, n_samples=1000, kernel_width=None):
    d = len(x)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # 1. Sample perturbations around x (normal noise scaled by feature std)
    Z = x + np.random.randn(n_samples, d) * std
    Z[0] = x                                    # first sample = original

    # 2. Predictions of the black-box model
    preds = model(Z)

    # 3. Distance in standardized space + exponential kernel
    Z_scaled = (Z - mean) / std
    x_scaled = (x - mean) / std
    distance = np.sqrt(np.sum((Z_scaled - x_scaled) ** 2, axis=1))
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(d)
    weights = np.exp(-(distance ** 2) / kernel_width ** 2)

    # 4. Weighted linear regression (ridge) on standardized features
    Zb = np.hstack([np.ones((n_samples, 1)), Z_scaled])   # add intercept
    ZtW = Zb.T * weights
    coef = np.linalg.solve(ZtW @ Zb + 1e-3 * np.eye(d + 1), ZtW @ preds)

    intercept = coef[0]
    importance = coef[1:]                       # local effect of each feature
    return intercept, importance


# ================= RUN =================
print("P(class 1) for x =", round(model(x.reshape(1, -1))[0], 4))
intercept, importance = lime_tabular(x, X_train, model)
print("Intercept:", round(intercept, 4))
print("Feature importances:")
for i in np.argsort(-np.abs(importance)):
    print("  {:>3s} : {:+.4f}".format(feature_names[i], importance[i]))


# ================= LIBRARY VERSION =================
# pip install lime
try:
    from lime.lime_tabular import LimeTabularExplainer

    def predict_proba(X):
        """lime needs probabilities for every class -> (n, n_classes)"""
        p = model(X)
        return np.column_stack([1 - p, p])

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names,
                                     class_names=["0", "1"], mode="classification",
                                     discretize_continuous=False)
    exp = explainer.explain_instance(x, predict_proba, num_features=3, num_samples=1000)
    print("\n[library] lime feature weights:")
    for name, weight in exp.as_list():
        print(f"  {name:>3s} : {weight:+.4f}")
    # with discretize_continuous=True you get rules like "f0 > 0.75"
except ImportError:
    print("\n[library] not installed -> pip install lime")


# ======================================================================
# THEORY: LIME for Tabular data
# ======================================================================
# Goal: explain ONE prediction of a black-box model on tabular data with a
#       local weighted linear model.
#
# Difference from image/text: we cannot "switch off" a number, so we
#   SAMPLE new points around x using the training data statistics.
#
# Steps:
#   1. Compute mean and std of each feature from training data.
#   2. Sample N perturbed points:  z = x + noise * std  (normal noise).
#      (The library version samples around the training mean and/or
#       discretizes features into quartiles.)
#   3. Get black-box predictions f(z).
#   4. Standardize features, compute Euclidean distance D(x, z) and the
#      kernel weight:  pi_x(z) = exp( -D^2 / sigma^2 ),
#      default sigma = 0.75 * sqrt(number of features).
#   5. Fit weighted ridge regression g(z) = w0 + sum_k w_k z_k.
#   6. w_k = local importance of feature k (sign = direction).
#
#   + Model-agnostic, simple, shows direction of each feature locally.
#   - Kernel width choice changes the result a lot.
#   - Sampling ignores feature correlation; explanations can be unstable.
#   - Only LOCALLY faithful (valid near x, not globally).
# ======================================================================
