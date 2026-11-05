import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
X = np.random.rand(5, 3)              # few samples so we can print each line


def model(X):
    return 3 * X[:, 0] + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 2]


# ================= IMPLEMENT =================
def ice(model, X, feature, n_grid=6):
    # 1. Grid of values for the chosen feature
    grid = np.linspace(X[:, feature].min(), X[:, feature].max(), n_grid)

    # 2. One curve per sample (rows = samples, cols = grid values)
    curves = np.zeros((X.shape[0], n_grid))
    for i in range(X.shape[0]):
        for g, v in enumerate(grid):
            x = X[i].copy()
            x[feature] = v
            curves[i, g] = model(x.reshape(1, -1))[0]

    return grid, curves


def centered_ice(curves):
    # c-ICE: subtract each curve's value at the first grid point
    return curves - curves[:, [0]]


# ================= RUN =================
feature = 0
grid, curves = ice(model, X, feature)
print("grid:", np.round(grid, 2))
print("\nICE curves (one row per sample):")
print(np.round(curves, 3))
print("\nCentered ICE:")
print(np.round(centered_ice(curves), 3))
print("\nPDP = mean of ICE curves:")
print(np.round(curves.mean(axis=0), 3))


# ================= LIBRARY VERSION =================
# pip install scikit-learn   (needs a fitted sklearn estimator)
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay

    X_big = np.random.rand(200, 3)
    est = GradientBoostingRegressor(random_state=0).fit(X_big, model(X_big))
    res = partial_dependence(est, X, features=[feature], kind="individual", grid_resolution=6)
    print("\n[library] sklearn ICE curves:")
    print(np.round(res["individual"][0], 3))            # (n_samples, n_grid)
    # plot:  PartialDependenceDisplay.from_estimator(est, X, [feature], kind="both", centered=True)
    #        kind="individual" -> ICE only, kind="both" -> ICE + PDP, centered=True -> c-ICE
except ImportError:
    print("\n[library] not installed -> pip install scikit-learn")


# ======================================================================
# THEORY: ICE (Individual Conditional Expectation)
# ======================================================================
# Goal: LOCAL version of PDP -> one curve PER INSTANCE showing how that
#       instance's prediction changes when one feature changes.
#
# Formula:
#   ICE_i(v) = f(v, x_-j^i)      (for each row i, vary only feature j)
#   PDP(v)   = mean over i of ICE_i(v)
#
# Steps:
#   1. Pick a grid of values for feature j.
#   2. For each instance i and each grid value v:
#        copy x_i, set x_ij = v, predict.
#   3. Plot one line per instance.
#
# Centered ICE (c-ICE): subtract each curve's value at a reference point
#   (usually the first grid value) so all curves start at 0 -> easier to
#   compare shapes.
#
#   + Reveals heterogeneity / interactions that PDP averages away
#     (if curves are not parallel -> feature interacts with others).
#   - Same independence assumption as PDP.
#   - Plot gets crowded with many instances.
# ======================================================================
