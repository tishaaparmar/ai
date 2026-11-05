import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
X = np.random.rand(200, 3)            # 200 samples, 3 features


def model(X):
    return 3 * X[:, 0] + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 2]


# ================= IMPLEMENT =================
def accumulated_local_effects(model, X, feature, n_bins=10):
    x = X[:, feature]

    # 1. Bin edges using quantiles (equal number of points per bin)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))

    local_effects = []
    counts = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]

        # 2. Points that fall inside this bin
        if b == 0:
            in_bin = (x >= lo) & (x <= hi)
        else:
            in_bin = (x > lo) & (x <= hi)
        Xb = X[in_bin]

        if len(Xb) == 0:
            local_effects.append(0.0)
            counts.append(0)
            continue

        # 3. Move those points to the lower and upper edge of the bin
        X_lo = Xb.copy()
        X_lo[:, feature] = lo
        X_hi = Xb.copy()
        X_hi[:, feature] = hi

        # 4. Local effect = average prediction difference
        local_effects.append(np.mean(model(X_hi) - model(X_lo)))
        counts.append(len(Xb))

    # 5. Accumulate (cumulative sum), ALE starts at 0 at the first edge
    ale = np.concatenate([[0.0], np.cumsum(local_effects)])

    # 6. Center so that the average effect is 0
    bin_centers_ale = (ale[:-1] + ale[1:]) / 2
    counts = np.array(counts)
    ale = ale - np.sum(bin_centers_ale * counts) / np.sum(counts)

    return edges, ale


# ================= RUN =================
for f in range(X.shape[1]):
    edges, ale = accumulated_local_effects(model, X, feature=f)
    print(f"Feature {f}")
    print("  grid :", np.round(edges, 2))
    print("  ALE  :", np.round(ale, 3))


# ================= LIBRARY VERSION =================
# pip install alibi
try:
    from alibi.explainers import ALE, plot_ale

    ale_lib = ALE(model, feature_names=["f0", "f1", "f2"])
    exp = ale_lib.explain(X)
    print("\n[library] alibi ALE for feature 0")
    print("  grid :", np.round(exp.feature_values[0], 2))
    print("  ALE  :", np.round(exp.ale_values[0].ravel(), 3))
    # plot_ale(exp)
except ImportError:
    print("\n[library] not installed -> pip install alibi")


# ======================================================================
# THEORY: ALE (Accumulated Local Effects)
# ======================================================================
# Goal: show how a feature affects the prediction ON AVERAGE (global),
#       like PDP, but CORRECT when features are correlated.
#
# Problem with PDP: it sets feature = v for ALL rows, creating unrealistic
#   points (e.g. tiny house with 10 rooms) when features are correlated.
#
# ALE idea: only look at rows that ACTUALLY lie near a value, and measure
#   the LOCAL CHANGE (difference) in prediction, not the absolute value.
#
# Steps:
#   1. Split feature x_j into bins using quantiles (z_0, z_1, ..., z_K).
#   2. For each bin k, take the rows whose x_j falls inside that bin.
#   3. For those rows compute: f(x_j = z_k) - f(x_j = z_{k-1})
#      (move only x_j to the upper and lower edge, keep others same).
#   4. Average these differences -> local effect of bin k.
#   5. Accumulate (cumulative sum) over bins -> ALE curve.
#   6. Center the curve so its mean is 0.
#
# Formula (uncentered):
#   ALE_j(x) = sum_{k=1}^{k(x)}  (1/n_k) * sum_{i in bin k}
#              [ f(z_k, x_{-j}^i) - f(z_{k-1}, x_{-j}^i) ]
#
# Interpretation: ALE(x) = effect of x_j at value x compared to the
#   average prediction.
#   + Unbiased with correlated features, faster than PDP.
#   - Harder to explain; depends on number of bins.
# ======================================================================
