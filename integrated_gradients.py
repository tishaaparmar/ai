import numpy as np

# ================= GIVEN (outside code) =================
w = np.array([1.5, -2.0, 0.5, 3.0])
x = np.array([1.0, 0.5, 2.0, 0.8])
baseline = np.zeros_like(x)


def model(x):
    return 1 / (1 + np.exp(-(w @ x)))          # sigmoid(w.x)


def gradient(x):
    """d(model)/dx"""
    s = model(x)
    return s * (1 - s) * w


# ================= IMPLEMENT =================
def integrated_gradients(x, baseline, gradient, steps=50):
    total_grad = np.zeros_like(x)

    # 1. Walk on the straight line from baseline to x
    for k in range(1, steps + 1):
        alpha = k / steps
        x_step = baseline + alpha * (x - baseline)
        # 2. Accumulate gradients along the path
        total_grad += gradient(x_step)

    # 3. Average gradient (Riemann sum approximation of the integral)
    avg_grad = total_grad / steps

    # 4. Scale by (input - baseline)
    return (x - baseline) * avg_grad


# ================= RUN =================
ig = integrated_gradients(x, baseline, gradient)
print("IG attributions:", np.round(ig, 4))
print("Sum of IG        :", round(ig.sum(), 4))
print("f(x) - f(baseline):", round(model(x) - model(baseline), 4), "(completeness check)")


# ================= LIBRARY VERSION =================
# pip install captum torch
try:
    import torch
    from captum.attr import IntegratedGradients

    w_t = torch.tensor(w, dtype=torch.float32)

    def torch_model(inp):                               # same model in torch
        return torch.sigmoid(inp @ w_t)

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)     # (1, 4)
    ig_lib = IntegratedGradients(torch_model)
    attr, delta = ig_lib.attribute(x_t, baselines=torch.zeros_like(x_t),
                                   n_steps=50, return_convergence_delta=True)
    print("\n[library] captum IG:", np.round(attr.detach().numpy()[0], 4))
    print("[library] convergence delta:", round(delta.item(), 6))
    # for a classifier with many outputs: ig_lib.attribute(x_t, target=class_idx)
except ImportError:
    print("\n[library] not installed -> pip install captum torch")


# ======================================================================
# THEORY: IG (Integrated Gradients)
# ======================================================================
# Goal: attribute the prediction of a differentiable model (neural net)
#       to each input feature.
#
# Problem with plain gradients: SATURATION - when a feature is already
#   strong, the gradient becomes ~0 even though the feature matters.
#
# Idea: choose a BASELINE x' (black image / zero vector / empty text) and
#   integrate the gradient along the straight path from x' to x.
#
# Formula:
#   IG_i(x) = (x_i - x'_i) * integral_{a=0}^{1}  df(x' + a(x - x')) / dx_i  da
#
# Riemann sum approximation (what the code does), m steps:
#   IG_i(x) ~= (x_i - x'_i) * (1/m) * sum_{k=1}^{m}
#              df( x' + (k/m)(x - x') ) / dx_i
#
# Axioms satisfied:
#   - Completeness:  sum_i IG_i(x) = f(x) - f(x')
#   - Sensitivity:   if x and x' differ in one feature and the predictions
#                    differ, that feature gets non-zero attribution.
#   - Implementation invariance: same function -> same attributions.
#
#   + Theoretically grounded, only needs gradients (fast).
#   - Result depends on the choice of baseline.
#   - Needs a differentiable model.
# ======================================================================
