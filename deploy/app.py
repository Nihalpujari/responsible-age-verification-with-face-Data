"""
Age Verification for Age-Restricted Sales — Streamlit app.

German Jugendschutzgesetz § 9: 16+ for beer/wine, 18+ for spirits.

The model never decides the sale on its own. It produces a distribution over
9 age bands; the app computes P(age >= legal threshold) and returns one of
three outcomes, the third being an explicit hand-off to a human — the
concrete implementation of EU AI Act Article 14 in this system.

Deployed on Streamlit Community Cloud. Local run:  streamlit run app.py
"""
import base64
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from responsible_ai_utils import p_at_least, decide, AGE_GROUP_LABELS

# ---------------------------------------------------------------- config ----
HERE = Path(__file__).parent

PRODUCTS = {
    "Beer / Wine / Sparkling": 16,
    "Spirits / Liquor": 18,
}
P_APPROVE, P_REJECT = 0.95, 0.05
IMG_SIZE = 224

# Share of the 10-19 band at or above each threshold, from the training
# age distribution. Falls back to these constants when the CSVs are absent
# (they are not shipped with the deployed app).
BAND_FRAC = {16: 0.40, 18: 0.20}

# Bands with per-class recall < 40% in the audit — surfaced as a warning.
LOW_RELIABILITY = {"30-39", "40-49", "50-59"}

st.set_page_config(page_title="Age Verification", page_icon="🍺",
                   layout="centered", initial_sidebar_state="collapsed")


# ----------------------------------------------------------------- style ----
@st.cache_data
def _bg_data_uri():
    p = HERE / "background.jpg"
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def inject_css():
    bg = _bg_data_uri()
    layer = (
        f"linear-gradient(180deg, rgba(8,10,20,.82) 0%, rgba(8,10,20,.93) 55%,"
        f" rgba(8,10,20,.97) 100%), url('data:image/jpeg;base64,{bg}')"
        if bg else
        "radial-gradient(1200px 600px at 20% -10%, #24304d 0%, #0b0e18 60%)"
    )
    st.markdown(f"""
    <style>
      .stApp {{
        background: {layer};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
      }}
      .block-container {{ padding-top: 2.2rem; max-width: 940px; }}
      .stApp, .stApp p, .stApp label, .stApp span {{ color: #e8ecf5; }}

      .hero h1 {{
        font-size: 2.3rem; font-weight: 700; letter-spacing: -.02em;
        margin: 0 0 .3rem 0; color: #fff;
      }}
      .hero p {{ color: #aab3c7; margin: 0; font-size: .95rem; }}

      .card {{
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
      }}

      .verdict {{
        border-radius: 18px; padding: 1.5rem 1.6rem; margin: .4rem 0 1rem 0;
        border: 1px solid; backdrop-filter: blur(14px);
      }}
      .verdict .tag {{
        font-size: .74rem; font-weight: 700; letter-spacing: .14em;
        text-transform: uppercase; opacity: .85;
      }}
      .verdict .headline {{
        font-size: 1.85rem; font-weight: 700; margin: .35rem 0 .5rem 0;
        line-height: 1.15;
      }}
      .verdict .sub {{ font-size: .93rem; opacity: .88; }}

      .v-ok   {{ background: rgba(22,163,74,.16);  border-color: rgba(74,222,128,.45); }}
      .v-ok   .headline, .v-ok .tag   {{ color: #6ee7a8; }}
      .v-no   {{ background: rgba(220,38,38,.16);  border-color: rgba(248,113,113,.45); }}
      .v-no   .headline, .v-no .tag   {{ color: #fca5a5; }}
      .v-warn {{ background: rgba(217,119,6,.17);  border-color: rgba(251,191,36,.48); }}
      .v-warn .headline, .v-warn .tag {{ color: #fcd34d; }}

      .meter {{
        height: 9px; border-radius: 99px; overflow: hidden;
        background: rgba(255,255,255,.13); margin: .8rem 0 .35rem 0;
      }}
      .meter > div {{ height: 100%; border-radius: 99px; }}

      .zones {{
        display: flex; justify-content: space-between;
        font-size: .72rem; color: #97a1b8; letter-spacing: .03em;
      }}

      .brow {{ display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }}
      .brow .lbl {{ width: 62px; font-size: .82rem; color: #c3cbdd; }}
      .brow .bar {{ flex: 1; height: 7px; background: rgba(255,255,255,.09);
                    border-radius: 99px; overflow: hidden; }}
      .brow .bar > div {{ height: 100%; background: #7c9cf5; border-radius: 99px; }}
      .brow .val {{ width: 46px; text-align: right; font-size: .78rem;
                    color: #9aa5bd; font-variant-numeric: tabular-nums; }}

      .foot {{ color: #7d879e; font-size: .78rem; line-height: 1.6; }}
      .foot strong {{ color: #a9b3c9; }}
      .foot a {{ color: #7c9cf5; text-decoration: none; }}

      [data-testid="stImage"] img {{ border-radius: 12px; }}
      div[data-testid="stCameraInput"] button,
      section[data-testid="stFileUploaderDropzone"] {{ border-radius: 12px; }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ----------------------------------------------------------------- model ----
_KWARGS_PATCHED = False


def _patch_keras_compat():
    """Keras 3's image-preprocessing API keeps changing keyword arguments across
    point releases (`value_range`, `bounding_box_format`, ...). Saved models
    carry those kwargs; a newer Keras rejects them. `custom_objects=` does not
    help here because the saved config has `module: 'keras.layers'` and Keras
    looks the class up directly, bypassing any subclass we supply.

    Fix: monkey-patch `__init__` on the real Keras classes to silently drop
    stale kwargs. Applied once, before any load_model call."""
    global _KWARGS_PATCHED
    if _KWARGS_PATCHED:
        return
    from tensorflow.keras import layers as kl

    stale = ("value_range", "bounding_box_format")

    for name in dir(kl):
        if not name.startswith("Random"):
            continue
        cls = getattr(kl, name, None)
        if not isinstance(cls, type):
            continue
        orig_init = cls.__init__

        def make(orig):
            def new_init(self, *args, **kwargs):
                for k in stale:
                    kwargs.pop(k, None)
                return orig(self, *args, **kwargs)
            return new_init

        cls.__init__ = make(orig_init)

    _KWARGS_PATCHED = True


@st.cache_resource(show_spinner="Loading model (first request only)…")
def load_model():
    import tensorflow as tf
    _patch_keras_compat()

    candidates = sorted(HERE.glob("*.keras"))
    if not candidates:
        return None, False, None
    path = candidates[0]
    m = tf.keras.models.load_model(path, compile=False)

    def has_rescaling(model):
        """v7 bakes Rescaling in and therefore expects raw [0,255]."""
        for l in model.layers:
            if isinstance(l, tf.keras.layers.Rescaling):
                return True
            if isinstance(l, tf.keras.Model) and has_rescaling(l):
                return True
        return False

    return m, has_rescaling(m), path.name


@st.cache_resource
def load_face_detector():
    import cv2
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(str(xml))
    return None if clf.empty() else clf


def crop_face(pil_img):
    """Return (cropped_image, found_bool). Falls back to the full frame."""
    try:
        import cv2
    except ImportError:
        return pil_img, False
    clf = load_face_detector()
    if clf is None:
        return pil_img, False

    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                 minSize=(60, 60))
    if len(faces) == 0:
        return pil_img, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.28 * max(w, h))
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return Image.fromarray(rgb[y0:y1, x0:x1]), True


def predict(pil_img, model, needs_raw):
    x = np.asarray(
        pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        np.float32)
    if not needs_raw:
        x = x / 255.0
    return model.predict(x[None], verbose=0)[0]


# -------------------------------------------------------------------- UI ----
st.markdown(
    '<div class="hero"><h1>Age Verification</h1>'
    '<p>Point-of-sale check for age-restricted products · '
    'Jugendschutzgesetz § 9</p></div>', unsafe_allow_html=True)
st.write("")

model, NEEDS_RAW, model_name = load_model()

if model is None:
    st.error(
        "**No model file found.** The `.keras` weights have not been included "
        "in this deployment. Please contact the repository owner.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    product = st.radio("Product being purchased", list(PRODUCTS),
                       captions=[f"legal age {v}" for v in PRODUCTS.values()])
with right:
    source = st.radio("Capture method", ["Upload photo", "Use camera"])

THRESHOLD = PRODUCTS[product]
FRAC = BAND_FRAC[THRESHOLD]

img = None
if source == "Upload photo":
    up = st.file_uploader("Customer photo", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    if up:
        img = Image.open(up)
else:
    shot = st.camera_input("Customer photo", label_visibility="collapsed")
    if shot:
        img = Image.open(shot)

if img is None:
    st.markdown(
        '<div class="card foot">Select a product, then upload or capture a '
        'photo of the customer.<br><br><strong>This is a course demonstrator, '
        'not a product.</strong> The underlying model is roughly 54% accurate '
        'at exact age bands and has measured accuracy gaps across race groups. '
        'It is built to study human oversight, not to replace it.</div>',
        unsafe_allow_html=True)
    st.stop()

face, found = crop_face(img)
probs = predict(face, model, NEEDS_RAW)
p_over = p_at_least(probs, FRAC)
outcome = decide(p_over, P_APPROVE, P_REJECT)
top = int(np.argmax(probs))
top_band = AGE_GROUP_LABELS[top]

STYLE = {
    "auto-clear":   ("v-ok",   "Approved",       "#4ade80",
                     f"Sale of {product.lower()} may proceed."),
    "auto-reject":  ("v-no",   "Refused",        "#f87171",
                     f"Customer is below the legal age of {THRESHOLD}."),
    "human-review": ("v-warn", "Human check required", "#fbbf24",
                     "The system is not confident enough to decide. "
                     "Escalate to a staff member for manual ID inspection."),
}
cls, headline, colour, sub = STYLE[outcome]

st.markdown(f"""
<div class="verdict {cls}">
  <div class="tag">{THRESHOLD}+ · {product}</div>
  <div class="headline">{headline}</div>
  <div class="sub">{sub}</div>
  <div class="meter"><div style="width:{p_over*100:.1f}%;background:{colour};"></div></div>
  <div class="zones">
    <span>refuse ≤ 5%</span>
    <span>P(age ≥ {THRESHOLD}) = <strong style="color:{colour}">{p_over:.1%}</strong></span>
    <span>approve ≥ 95%</span>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.35], gap="large")

with c1:
    st.image(face, width="stretch",
             caption="Detected face" if found else "No face detected — full frame used")
    if not found:
        st.warning("No face was detected. Accuracy degrades on uncropped "
                   "images; retake the photo closer to the customer.")

with c2:
    st.markdown("**Age band distribution**")
    rows = []
    hi = float(probs.max())
    for lbl, pr in zip(AGE_GROUP_LABELS, probs):
        w = 0 if hi == 0 else pr / hi * 100
        strong = "opacity:1" if lbl == top_band else "opacity:.55"
        rows.append(
            f'<div class="brow"><span class="lbl">{lbl}</span>'
            f'<span class="bar"><div style="width:{w:.1f}%;{strong}"></div></span>'
            f'<span class="val">{pr:.0%}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)

    if top_band == "10-19":
        st.info(
            "**10-19 straddles the legal age.** This band is never counted as "
            f"adult on its own — only the {FRAC:.0%} of it estimated to be "
            f"{THRESHOLD}+ contributes to the decision.")
    elif top_band in LOW_RELIABILITY:
        st.warning(
            f"**{top_band} is a low-reliability band** — under 40% per-class "
            "recall in the fairness audit. Treat this prediction with caution.")

st.write("")
st.markdown(f"""
<div class="card foot">
<strong>How the decision is made.</strong> The model outputs a probability for each of
9 age bands. P(age ≥ {THRESHOLD}) sums all bands above the threshold plus the
estimated {FRAC:.0%} of the 10-19 band that clears it. Approve at ≥95%, refuse at ≤5%,
and <strong>escalate to a human anywhere in between</strong> — EU AI Act Article 14.<br><br>
<strong>Known limits.</strong> ~54% exact-band accuracy; measured accuracy gaps across race
groups; trained on UTKFace, whose subjects did not consent to this use.
Not fit for real age verification.<br><br>
Model: <code>{model_name}</code> · input {'[0,255]' if NEEDS_RAW else '[0,1]'} (auto-detected)
· <a href="https://github.com/Nihalpujari/Face-Recognition">source</a>
</div>
""", unsafe_allow_html=True)
