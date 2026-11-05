import numpy as np

# ================= GIVEN (outside code) =================
np.random.seed(0)
text = "the movie was great and the acting was good but story was boring"

positive = {"great": 2.0, "good": 1.5}
negative = {"boring": -2.0, "bad": -1.5}


def model(sentence):
    """toy sentiment classifier -> P(positive)"""
    s = 0.0
    for w in sentence.split():
        s += positive.get(w, 0) + negative.get(w, 0)
    return 1 / (1 + np.exp(-s))


# ================= IMPLEMENT =================
def lime_text(text, model, n_samples=500, kernel_width=0.25):
    words = text.split()
    d = len(words)

    # 1. Random binary masks (1 = keep word, 0 = remove word)
    Z = np.random.randint(0, 2, size=(n_samples, d))
    Z[0] = 1                                   # original sentence

    # 2. Predictions on perturbed sentences
    preds = np.zeros(n_samples)
    for n in range(n_samples):
        sentence = " ".join(w for w, keep in zip(words, Z[n]) if keep)
        preds[n] = model(sentence)

    # 3. Weights: cosine distance to original + exponential kernel
    original = np.ones(d)
    cos_sim = (Z @ original) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(original) + 1e-10)
    distance = 1 - cos_sim
    weights = np.exp(-(distance ** 2) / kernel_width ** 2)

    # 4. Weighted linear regression (ridge)
    Zb = np.hstack([np.ones((n_samples, 1)), Z])
    ZtW = Zb.T * weights
    coef = np.linalg.solve(ZtW @ Zb + 1e-3 * np.eye(d + 1), ZtW @ preds)

    return words, coef[1:]


# ================= RUN =================
print("P(positive) =", round(model(text), 3))
words, importance = lime_text(text, model)
print("\nWord importances:")
for i in np.argsort(-np.abs(importance)):
    print(f"  {words[i]:>8s} : {importance[i]:+.3f}")


# ================= LIBRARY VERSION =================
# pip install lime
try:
    from lime.lime_text import LimeTextExplainer

    def classifier_fn(texts):
        """lime gives a list of strings -> must return probs (n, n_classes)"""
        p = np.array([model(t) for t in texts])
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    exp = explainer.explain_instance(text, classifier_fn, num_features=6, num_samples=500)
    print("\n[library] lime word weights:")
    for word, weight in exp.as_list():
        print(f"  {word:>8s} : {weight:+.3f}")
    # notebook: exp.show_in_notebook(text=True)
except ImportError:
    print("\n[library] not installed -> pip install lime")


# ======================================================================
# THEORY: LIME for Text
# ======================================================================
# Same idea as LIME for images, but the interpretable features are WORDS.
#
# Interpretable representation: binary vector z of length d (= number of
#   words). z_k = 1 -> word k is kept, 0 -> word k is removed.
#
# Steps:
#   1. Split the sentence into d words.
#   2. Generate N random binary masks z.
#   3. For each mask, rebuild the sentence with only the kept words and get
#      the classifier's prediction f(z).
#   4. Weight samples by closeness to the original sentence:
#          pi_x(z) = exp( -D(x,z)^2 / sigma^2 )   (cosine distance)
#   5. Fit weighted linear model g(z) = w0 + sum_k w_k z_k.
#   6. w_k > 0 -> word pushes towards the class,
#      w_k < 0 -> word pushes against the class.
#
# Example: "great", "good" get positive weights; "boring" negative.
#
#   + Very readable explanation (highlight words).
#   - Removing words can produce ungrammatical text;
#     ignores word order.
# ======================================================================
