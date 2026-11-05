import numpy as np
from itertools import combinations
from math import factorial

# ================= GIVEN (outside code) =================
np.random.seed(0)
background = np.random.rand(50, 3)     # background data (for "missing" features)
x = np.array([0.9, 0.2, 0.7])          # instance to explain


def model(X):
    return 3 * X[:, 0] + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 2]


# ================= IMPLEMENT =================
def value_function(model, x, background, S):
    """v(S): features in S fixed to x, others taken from background, then averaged"""
    X = background.copy()
    X[:, S] = x[S]
    return np.mean(model(X))


def shapley_values(model, x, background):
    n = len(x)
    phi = np.zeros(n)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        # loop over every subset S of the other features
        for size in range(n):
            for S in combinations(others, size):
                S = list(S)
                # Shapley weight = |S|! (n-|S|-1)! / n!
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                # marginal contribution of feature i
                gain = value_function(model, x, background, S + [i]) - value_function(model, x, background, S)
                phi[i] += weight * gain

    return phi


# ================= RUN =================
phi = shapley_values(model, x, background)
base = value_function(model, x, background, [])
print("Base value E[f(X)]:", round(base, 4))
print("SHAP values       :", np.round(phi, 4))
print("base + sum(phi)   :", round(base + phi.sum(), 4))
print("f(x)              :", round(model(x.reshape(1, -1))[0], 4), "(should match)")


# ================= LIBRARY VERSION =================
# pip install shap
# (this file is NOT named shap.py on purpose: that name would hide the library)
try:
    import shap

    explainer = shap.KernelExplainer(model, background)        # model-agnostic
    shap_vals = explainer.shap_values(x)
    print("\n[library] shap KernelExplainer")
    print("  base value :", round(float(explainer.expected_value), 4))
    print("  SHAP values:", np.round(shap_vals, 4))
    # tree models : shap.TreeExplainer(tree_model).shap_values(X)
    # neural nets : shap.DeepExplainer(net, background_tensor)
    # plots       : shap.summary_plot(vals, X), shap.force_plot(...), shap.plots.waterfall(...)
except ImportError:
    print("\n[library] not installed -> pip install shap")


# ======================================================================
# THEORY: SHAP (SHapley Additive exPlanations)
# ======================================================================
# Goal: fairly divide the prediction among the features, using Shapley
#       values from cooperative game theory.
#   players = features,  payout = prediction - average prediction.
#
# Value function: v(S) = E[ f(x) | features in S fixed to their values ]
#   (approximated by fixing S to x and taking the others from background
#    data, then averaging the predictions).
#
# Shapley value of feature i (average marginal contribution over all
# subsets S that do not contain i):
#   phi_i = sum_{S subset F\{i}}  |S|! (|F|-|S|-1)! / |F|!
#           * [ v(S U {i}) - v(S) ]
#
# Additive explanation:
#   f(x) = phi_0 + sum_i phi_i        (phi_0 = E[f(X)], base value)
#
# Properties (why Shapley is "fair"):
#   - Efficiency: contributions add up to f(x) - E[f(X)].
#   - Symmetry:   equal contribution -> equal value.
#   - Dummy:      feature that never changes v gets 0.
#   - Additivity: for combined models, values add.
#
# Exact computation needs 2^n subsets -> exponential. Approximations:
#   KernelSHAP (model-agnostic, LIME-like weighted regression),
#   TreeSHAP (fast exact for trees), DeepSHAP (neural nets).
#
#   + Solid theory, local AND global (mean |phi| = global importance).
#   - Expensive; uses background data (independence assumption).
# ======================================================================
