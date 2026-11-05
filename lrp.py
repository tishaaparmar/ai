import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
x = np.array([1.0, 0.5, -0.3, 0.8])
W1 = np.random.randn(4, 3)     # input(4) -> hidden(3)
b1 = np.zeros(3)
W2 = np.random.randn(3, 1)     # hidden(3) -> output(1)
b2 = np.zeros(1)


def relu(z):
    return np.maximum(z, 0)


def forward(x):
    a0 = x
    a1 = relu(a0 @ W1 + b1)
    out = a1 @ W2 + b2
    return a0, a1, out


# ================= IMPLEMENT =================
def lrp_layer(a, W, b, R_out, eps=1e-6):
    """LRP epsilon rule:  R_i = sum_j  a_i * w_ij / (z_j + eps) * R_j"""
    z = a @ W + b                                   # pre-activations of next layer
    z = z + eps * np.where(z >= 0, 1, -1)           # stabilizer (avoid /0)
    s = R_out / z                                   # (out,)
    c = W @ s                                       # (in,)  = sum_j w_ij * s_j
    return a * c                                    # relevance of inputs


def lrp(x):
    # 1. Forward pass, store activations
    a0, a1, out = forward(x)

    # 2. Relevance at output = output score
    R2 = out

    # 3. Propagate relevance backwards layer by layer
    R1 = lrp_layer(a1, W2, b2, R2)
    R0 = lrp_layer(a0, W1, b1, R1)
    return out, R1, R0


# ================= RUN =================
out, R1, R0 = lrp(x)
print("Output          :", np.round(out, 4))
print("Hidden relevance:", np.round(R1, 4), " sum =", round(R1.sum(), 4))
print("Input relevance :", np.round(R0, 4), " sum =", round(R0.sum(), 4))
print("(relevance is conserved: sums equal the output)")


# ================= LIBRARY VERSION =================
# pip install captum torch
try:
    import torch
    import torch.nn as nn
    from captum.attr import LRP

    net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():                               # copy the same weights
        net[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float32))
        net[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        net[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float32))
        net[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    attr = LRP(net).attribute(x_t, target=0)            # default rule for Linear = epsilon
    print("\n[library] captum LRP input relevance:", np.round(attr.detach().numpy()[0], 4))
    # other library: zennit (rules: Epsilon, Gamma, AlphaBeta, ZBox)
except ImportError:
    print("\n[library] not installed -> pip install captum torch")


# ======================================================================
# THEORY: LRP (Layer-wise Relevance Propagation)
# ======================================================================
# Goal: explain a neural network's output by propagating the output score
#       ("relevance") BACKWARDS layer by layer down to the input features.
#
# Conservation principle: total relevance is kept the same at every layer
#   sum_i R_i(layer l) = sum_j R_j(layer l+1) = ... = f(x)
#
# Each neuron j gives its relevance R_j to the inputs i in proportion to
# how much each input contributed to j's pre-activation z_ij = a_i * w_ij.
#
# Basic rule (LRP-0):
#   R_i = sum_j  ( a_i * w_ij / sum_i' a_i' * w_i'j ) * R_j
#
# Epsilon rule (LRP-eps, used in code): add a small stabilizer so we never
# divide by ~0 and weak/noisy contributions get absorbed:
#   R_i = sum_j  a_i * w_ij / ( z_j + eps * sign(z_j) ) * R_j
#
# Other rules: LRP-gamma (favours positive weights), alpha-beta rule
#   (separates positive and negative contributions), z^B rule for pixels.
#
# Steps:
#   1. Forward pass, store activations a of every layer.
#   2. Set relevance of output layer = output score f(x).
#   3. Apply the rule layer by layer back to the input.
#   4. Input relevances R_i = heatmap (positive = evidence FOR the class).
#
#   + Fast (one backward pass), sharp heatmaps, conservation property.
#   - Needs access to model internals; results depend on chosen rules.
# ======================================================================
