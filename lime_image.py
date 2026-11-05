import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
image = np.random.rand(8, 8)

# superpixels: 4x4 grid of 2x2 blocks -> 16 segments, labels 0..15
segments = np.zeros((8, 8), dtype=int)
for i in range(8):
    for j in range(8):
        segments[i, j] = (i // 2) * 4 + (j // 2)


def model(img):
    """toy classifier: probability depends on the top-left 4x4 region"""
    return 1 / (1 + np.exp(-(img[:4, :4].sum() - 4)))


# ================= IMPLEMENT =================
def lime_image(image, segments, model, n_samples=500, kernel_width=0.25):
    n_seg = segments.max() + 1

    # 1. Random binary masks (1 = keep superpixel, 0 = turn off)
    Z = np.random.randint(0, 2, size=(n_samples, n_seg))
    Z[0] = 1                                   # first sample = original image

    # 2. Build perturbed images and get predictions
    preds = np.zeros(n_samples)
    for n in range(n_samples):
        img = image.copy()
        for s in range(n_seg):
            if Z[n, s] == 0:
                img[segments == s] = 0         # hide superpixel (black)
        preds[n] = model(img)

    # 3. Weight by closeness to original (cosine distance + exp kernel)
    original = np.ones(n_seg)
    cos_sim = (Z @ original) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(original) + 1e-10)
    distance = 1 - cos_sim
    weights = np.exp(-(distance ** 2) / kernel_width ** 2)

    # 4. Weighted linear regression (ridge):  (Z^T W Z + lam I)^-1 Z^T W y
    Zb = np.hstack([np.ones((n_samples, 1)), Z])      # add intercept
    ZtW = Zb.T * weights
    coef = np.linalg.solve(ZtW @ Zb + 1e-3 * np.eye(n_seg + 1), ZtW @ preds)

    return coef[1:]                            # importance of each superpixel


# ================= RUN =================
importance = lime_image(image, segments, model)
print("Superpixel importance (as 4x4 grid of segments):")
print(np.round(importance.reshape(4, 4), 3))
print("\nTop 4 superpixels:", np.argsort(importance)[::-1][:4])


# ================= LIBRARY VERSION =================
# pip install lime scikit-image
try:
    from lime import lime_image as lime_img

    def classifier_fn(images):
        """lime gives a batch of RGB images (n, H, W, 3) -> must return probs (n, n_classes)"""
        p = np.array([model(img[:, :, 0]) for img in images])
        return np.column_stack([1 - p, p])

    image_rgb = np.stack([image] * 3, axis=-1)          # lime expects 3 channels
    explainer = lime_img.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb, classifier_fn,
        top_labels=1, hide_color=0, num_samples=500,
        segmentation_fn=lambda img: segments)           # default: quickshift superpixels

    label = explanation.top_labels[0]
    print("\n[library] lime superpixel weights (segment, weight):")
    print(sorted(explanation.local_exp[label], key=lambda t: -t[1])[:4])
    temp, mask = explanation.get_image_and_mask(label, positive_only=True,
                                                num_features=4, hide_rest=False)
    # show: from skimage.segmentation import mark_boundaries; plt.imshow(mark_boundaries(temp, mask))
except ImportError:
    print("\n[library] not installed -> pip install lime scikit-image")


# ======================================================================
# THEORY: LIME for Images (Local Interpretable Model-agnostic Explanations)
# ======================================================================
# Goal: explain ONE prediction by fitting a simple (linear) model that
#       mimics the black box in the NEIGHBOURHOOD of that image.
#
# Interpretable representation: pixels are too many, so the image is split
#   into SUPERPIXELS (segments, e.g. via quickshift / SLIC). Each superpixel
#   is a binary feature: 1 = present, 0 = hidden (grey/black).
#
# Steps:
#   1. Segment image into d superpixels.
#   2. Generate N random binary vectors z in {0,1}^d.
#   3. For each z, build the perturbed image (hide superpixels with 0)
#      and get the black-box prediction f(z).
#   4. Weight each sample by similarity to the original (all-ones vector):
#          pi_x(z) = exp( -D(x,z)^2 / sigma^2 )   (D = cosine distance)
#   5. Fit weighted linear regression g(z) = w0 + sum_k w_k z_k.
#   6. Coefficients w_k = importance of superpixel k. Show top-k
#      superpixels as the explanation.
#
# LIME objective:
#   explanation(x) = argmin_g  L(f, g, pi_x) + Omega(g)
#   (L = locality-weighted loss, Omega = complexity of g)
#
#   + Model-agnostic, intuitive (shows "which parts of the image").
#   - Depends on segmentation, kernel width and random sampling
#     (explanations can be unstable).
# ======================================================================
