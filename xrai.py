import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
image = np.random.rand(8, 8)
Wimg = np.zeros((8, 8))
Wimg[1:4, 1:4] = 1.0          # model "looks at" this region
Wimg[5:7, 4:7] = 0.5          # and a bit at this region
baseline = np.zeros_like(image)


def model(img):
    return np.sum(Wimg * img ** 2)


def gradient(img):
    return 2 * Wimg * img


def integrated_gradients(img, baseline, steps=50):
    total = np.zeros_like(img)
    for k in range(1, steps + 1):
        total += gradient(baseline + (k / steps) * (img - baseline))
    return (img - baseline) * total / steps


def make_segments():
    """over-segmentation: 2x2 blocks AND 4x4 blocks (overlapping) -> list of masks"""
    masks = []
    for size in [2, 4]:
        for i in range(0, 8, size):
            for j in range(0, 8, size):
                m = np.zeros((8, 8), dtype=bool)
                m[i:i + size, j:j + size] = True
                masks.append(m)
    return masks


# ================= IMPLEMENT =================
def xrai(attr, segments):
    """
    Greedy region growing:
    repeatedly pick the segment with the highest attribution DENSITY
    (gain / new area) and add it to the selected region.
    """
    selected = np.zeros(attr.shape, dtype=bool)
    saliency = np.zeros(attr.shape)
    remaining = list(range(len(segments)))
    order = []

    while remaining and not selected.all():
        best, best_gain = None, -np.inf
        for s in remaining:
            new_pixels = segments[s] & ~selected          # only pixels not yet chosen
            area = new_pixels.sum()
            if area == 0:
                continue
            gain = attr[new_pixels].sum() / area          # attribution per pixel
            if gain > best_gain:
                best, best_gain = s, gain

        if best is None:
            break
        new_pixels = segments[best] & ~selected
        saliency[new_pixels] = best_gain                  # region gets its gain value
        selected |= new_pixels
        order.append(best)
        remaining.remove(best)

    return saliency, order


# ================= RUN =================
attr = integrated_gradients(image, baseline)           # step 1: pixel attributions
segments = make_segments()                              # step 2: segmentation
saliency, order = xrai(attr, segments)                  # step 3: rank regions

print("Pixel attributions (IG):")
print(np.round(attr, 2))
print("\nXRAI region saliency:")
print(np.round(saliency, 2))
print("\nSegments in order of importance (first 5):", order[:5])


# ================= LIBRARY VERSION =================
# pip install saliency scikit-image   (Google PAIR saliency library)
try:
    import saliency.core as saliency

    def call_model_function(images, call_model_args=None, expected_keys=None):
        """saliency gives a batch (n, H, W, C) -> return gradients of the score"""
        grads = np.array([gradient(img[:, :, 0])[:, :, None] for img in images])
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: grads}

    img_3d = image[:, :, None]                          # (H, W, C)
    xrai_lib = saliency.XRAI()
    xrai_map = xrai_lib.GetMask(img_3d, call_model_function,
                                segments=segments)      # default: Felzenszwalb segments
    print("\n[library] saliency XRAI map:")
    print(np.round(xrai_map, 2))
except ImportError:
    print("\n[library] not installed -> pip install saliency scikit-image")


# ======================================================================
# THEORY: XRAI (eXplanation with Ranked Area Integrals)
# ======================================================================
# Goal: produce REGION-based saliency (which areas of the image matter)
#       instead of noisy pixel-level maps. Built on Integrated Gradients.
#
# Steps:
#   1. Compute pixel attributions with Integrated Gradients
#      (original paper uses two baselines: black and white image).
#   2. Over-segment the image into many regions at multiple scales
#      (paper: Felzenszwalb segmentation with different parameters,
#       here: 2x2 and 4x4 blocks). Regions can overlap.
#   3. Greedy region selection: start with an empty mask; at each step
#      add the region with the highest ATTRIBUTION DENSITY:
#          gain(region) = sum of attribution of its NEW pixels / new area
#   4. Repeat until the whole image is covered (or an area budget).
#   5. The order in which regions are added = ranking; each region gets
#      a saliency value (its gain) -> final region-level heatmap.
#
# Why: pixel attributions (IG, gradients) are scattered and noisy; humans
#   understand "this object region" better. XRAI groups pixels into
#   meaningful regions and ranks them.
#
#   + Cleaner, more human-readable maps; better on insertion/deletion tests.
#   - Slower (IG + segmentation + greedy search); depends on segmentation.
# ======================================================================
