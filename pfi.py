import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
X = np.random.rand(200, 3)
y = 3 * X[:, 0] + X[:, 1] ** 2 + 0.1 * np.random.randn(200)   # feature 2 is useless


def model(X):
    return 3 * X[:, 0] + X[:, 1] ** 2


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ================= IMPLEMENT =================
def permutation_feature_importance(model, X, y, n_repeats=10):
    # 1. Baseline error on original data
    base_error = mse(y, model(X))

    importance = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        increases = []
        for _ in range(n_repeats):
            # 2. Shuffle column j (breaks its link with y)
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            # 3. Error increase = importance
            increases.append(mse(y, model(X_perm)) - base_error)
        importance[j] = np.mean(increases)

    return base_error, importance


# ================= RUN =================
base, imp = permutation_feature_importance(model, X, y)
print("Baseline MSE:", round(base, 4))
for j, v in enumerate(imp):
    print(f"Feature {j}: importance = {v:.4f}")


# ================= LIBRARY VERSION =================
# pip install scikit-learn   (needs a fitted sklearn estimator)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    est = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)
    result = permutation_importance(est, X, y, n_repeats=10, random_state=0,
                                    scoring="neg_mean_squared_error")
    print("\n[library] sklearn permutation_importance:")
    for j in range(X.shape[1]):
        print(f"Feature {j}: {result.importances_mean[j]:.4f} +/- {result.importances_std[j]:.4f}")
except ImportError:
    print("\n[library] not installed -> pip install scikit-learn")


# ======================================================================
# THEORY: PFI (Permutation Feature Importance)
# ======================================================================
# Goal: GLOBAL importance of each feature = how much the model's error
#       increases when that feature's information is destroyed.
#
# Idea: randomly SHUFFLE one column. This breaks the relation between the
#   feature and the target but keeps its distribution. If the model relied
#   on it, error goes up a lot; if not, error barely changes.
#
# Steps (Breiman; Fisher, Rudin & Dominici):
#   1. Compute baseline error e_orig = L(y, f(X)).
#   2. For each feature j:
#        a. Permute column j -> X_perm
#        b. e_perm = L(y, f(X_perm))
#        c. Importance FI_j = e_perm - e_orig   (or ratio e_perm / e_orig)
#      Repeat several times and average (shuffling is random).
#   3. Sort features by FI.
#
#   + Model-agnostic, no retraining, easy to understand.
#   + Takes all interactions into account.
#   - Needs true labels y.
#   - Correlated features: shuffling makes unrealistic rows, and importance
#     gets split between correlated features.
#   - Results vary due to randomness.
# ======================================================================
