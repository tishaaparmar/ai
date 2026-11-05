import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
image = np.random.rand(8, 8)          # input image (8x8)
filters = np.random.randn(4, 3, 3)    # 4 conv filters (3x3)
W = np.random.randn(4, 6, 6)          # class weights on the feature maps


def feature_map(img):
    """conv (valid) + ReLU  ->  A of shape (K, 6, 6)  (each map is 2D 6x6)"""
    K = filters.shape[0]
    A = np.zeros((K, 6, 6))
    for k in range(K):
        for i in range(6):
            for j in range(6):
                A[k, i, j] = np.sum(img[i:i + 3, j:j + 3] * filters[k])
    return np.maximum(A, 0)


def score(A):
    """class score y_c"""
    return np.sum(W * A)


def grad_wrt_feature_map(A):
    """d(y_c)/dA  ->  shape (K, 6, 6)  (3D array)"""
    return W.copy()


# ================= IMPLEMENT =================
def grad_cam(A, grad):
    K = A.shape[0]

    # 1. Global average pooling of gradients -> one weight per feature map
    alpha = np.zeros(K)
    for k in range(K):
        alpha[k] = np.mean(grad[k])          # (1/Z) * sum_ij dy/dA_ij

    # 2. Weighted sum of feature maps
    cam = np.zeros((A.shape[1], A.shape[2]))  # 6x6
    for k in range(K):
        cam += alpha[k] * A[k]

    # 3. ReLU (keep only positive influence)
    cam = np.maximum(cam, 0)

    # 4. Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ================= RUN =================
A = feature_map(image)
grad = grad_wrt_feature_map(A)
print("Feature maps shape:", A.shape)
print("Gradient shape    :", grad.shape)
print("Class score       :", round(score(A), 3))

cam = grad_cam(A, grad)
print("\nGrad-CAM (6x6):")
print(np.round(cam, 2))


# ================= LIBRARY VERSION =================
# pip install grad-cam torch torchvision
try:
    import torch
    from torchvision.models import resnet18
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    net = resnet18(weights=None).eval()                 # any CNN
    target_layers = [net.layer4[-1]]                    # last conv layer
    input_tensor = torch.rand(1, 3, 224, 224)           # (batch, C, H, W)

    cam_lib = GradCAM(model=net, target_layers=target_layers)
    heatmap = cam_lib(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]
    print("\n[library] pytorch-grad-cam heatmap shape:", heatmap.shape)
    # overlay: from pytorch_grad_cam.utils.image import show_cam_on_image
    #          vis = show_cam_on_image(rgb_img_float01, heatmap, use_rgb=True)
except ImportError:
    print("\n[library] not installed -> pip install grad-cam torch torchvision")


# ======================================================================
# THEORY: Grad-CAM (Gradient-weighted Class Activation Mapping)
# ======================================================================
# Goal: show WHICH REGIONS of an image a CNN used to predict class c.
#
# Idea:
#   - The last conv layer has K feature maps A^k (each H x W, here 6x6).
#     They keep spatial info AND high-level meaning.
#   - The gradient dy_c/dA^k tells how much the class score y_c changes
#     if feature map k changes -> how important map k is for class c.
#
# Steps:
#   1. Forward pass -> feature maps A (K x H x W) and score y_c.
#   2. Backprop -> gradients dy_c/dA (K x H x W).
#   3. Neuron importance weights (global average pooling of gradients):
#          alpha_k = (1/Z) * sum_i sum_j  dy_c / dA^k_ij      (Z = H*W)
#   4. Weighted combination + ReLU:
#          L_GradCAM = ReLU( sum_k alpha_k * A^k )             (H x W)
#      ReLU keeps only features with POSITIVE influence on class c.
#   5. Normalize to [0,1] and upsample to image size -> heatmap overlay.
#
# Properties:
#   + Works on any CNN (no architecture change / retraining, unlike CAM).
#   + Class-discriminative (different heatmap per class).
#   - Low resolution (size of last conv layer, e.g. 6x6 / 14x14).
#   - Guided Grad-CAM = Grad-CAM x Guided Backprop for finer detail.
# ======================================================================
