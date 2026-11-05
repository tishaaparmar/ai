import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
X = np.random.rand(200, 3)


def model(X):
    return 3 * X[:, 0] + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 2]


# ================= IMPLEMENT =================
def partial_dependence(model, X, feature, n_grid=10):
    # 1. Grid of values for the chosen feature
    grid = np.linspace(X[:, feature].min(), X[:, feature].max(), n_grid)

    pdp = []
    for v in grid:
        # 2. Set the feature to v for EVERY sample (others unchanged)
        X_mod = X.copy()
        X_mod[:, feature] = v
        # 3. Average prediction over all samples
        pdp.append(np.mean(model(X_mod)))

    return grid, np.array(pdp)


# ================= RUN =================
for f in range(X.shape[1]):
    grid, pdp = partial_dependence(model, X, feature=f)
    print(f"Feature {f}")
    print("  grid :", np.round(grid, 2))
    print("  PDP  :", np.round(pdp, 3))


# ================= LIBRARY VERSION =================
# pip install scikit-learn   (needs a fitted sklearn estimator)
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay

    est = GradientBoostingRegressor(random_state=0).fit(X, model(X))
    res = partial_dependence(est, X, features=[0], kind="average",
                             grid_resolution=10, method="brute")    # brute = same as scratch
    print("\n[library] sklearn PDP for feature 0")
    print("  grid :", np.round(res["grid_values"][0], 2))
    print("  PDP  :", np.round(res["average"][0], 3))
    # plot:  PartialDependenceDisplay.from_estimator(est, X, features=[0, 1, 2])
    #        PartialDependenceDisplay.from_estimator(est, X, features=[(0, 2)])  # 2D PDP
except ImportError:
    print("\n[library] not installed -> pip install scikit-learn")


# ======================================================================
# THEORY: PDP (Partial Dependence Plot)
# ======================================================================
# Goal: GLOBAL, model-agnostic plot of the average effect of one (or two)
#       features on the model prediction.
#
# Formula:
#   PD_j(v) = E_{x_-j}[ f(v, x_-j) ]  ~=  (1/n) * sum_i f(v, x_-j^i)
#   (x_-j = all other features, taken from the real data)
#
# Steps:
#   1. Pick a grid of values v for feature j.
#   2. For each v: replace feature j with v in EVERY row.
#   3. Predict and average -> one point of the PDP.
#   4. Plot v vs average prediction.
#
# Interpretation: flat line -> feature not important; slope -> how the
#   prediction changes as the feature increases.
#
#   + Simple, intuitive, easy to implement.
#   - Assumes features are INDEPENDENT (creates unrealistic data points
#     when correlated -> use ALE instead).
#   - Averages hide heterogeneous effects (e.g. half go up, half go down
#     -> flat PDP). Use ICE to see them.
#   - Max ~2 features at a time.
# ======================================================================
