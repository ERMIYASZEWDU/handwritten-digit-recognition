"""Streamlit web app for handwritten digit recognition."""
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import tensorflow as tf
import keras

st.set_page_config(
    page_title="Digit Recogniser",
    page_icon="🔢",
    layout="centered",
)

IMG_SIZE = 96   # MobileNetV2 input size


@st.cache_resource
def load_model():
    return keras.models.load_model("handwritten_digit_cnn.keras",
                                   compile=False)


def otsu_threshold(arr: np.ndarray) -> int:
    hist  = np.bincount(arr.astype(np.uint8).ravel(), minlength=256)
    total = arr.size
    s_tot = float(np.dot(np.arange(256), hist))
    s_bg, w_bg, best, t_best = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        s_bg += t * hist[t]
        d = s_bg / w_bg - (s_tot - s_bg) / w_fg
        v = w_bg * w_fg * d * d
        if v > best:
            best, t_best = v, t
    return t_best


def preprocess(pil_img: Image.Image) -> np.ndarray:
    # 1. Grayscale + denoise
    gray = pil_img.convert("L").filter(ImageFilter.GaussianBlur(radius=1))
    arr  = np.array(gray)

    # 2. Otsu binarisation
    arr = np.where(arr > otsu_threshold(arr), 255, 0).astype(np.uint8)

    # 3. Digit = white on black
    if arr.mean() > 128:
        arr = 255 - arr

    # 4. Dilate to close stroke gaps
    arr = np.array(Image.fromarray(arr).filter(ImageFilter.MaxFilter(size=3)))

    # 5. Crop tightly to bounding box + 10% padding
    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = arr.shape
        pr = max(1, int((rmax - rmin) * 0.10))
        pc = max(1, int((cmax - cmin) * 0.10))
        arr = arr[max(0, rmin-pr):min(h, rmax+pr+1),
                  max(0, cmin-pc):min(w, cmax+pc+1)]

    # 6. Resize preserving aspect ratio, centre-pad to IMG_SIZE
    crop   = Image.fromarray(arr)
    crop.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    padded = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    padded.paste(crop, ((IMG_SIZE - crop.width)  // 2,
                        (IMG_SIZE - crop.height) // 2))

    # 7. Convert to 3-channel RGB (MobileNetV2 expects 3 channels)
    rgb = Image.merge("RGB", [padded, padded, padded])
    return np.expand_dims(np.array(rgb, dtype=np.float32) / 255.0, axis=0)


model = load_model()

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload a photo of a handwritten digit (0 – 9).")

uploaded = st.file_uploader("Choose image…", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    pil_img = Image.open(io.BytesIO(uploaded.read()))

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    proc = preprocess(pil_img)

    with col2:
        # Show grayscale version of preprocessed image
        preview = (proc[0, :, :, 0] * 255).astype(np.uint8)
        st.image(preview, caption="After preprocessing", use_container_width=True)

    with st.spinner("Predicting…"):
        probs = model.predict(proc, verbose=0)[0]

    digit = int(np.argmax(probs))
    conf  = float(probs[digit]) * 100

    st.success(f"**Predicted digit: {digit}** — {conf:.1f}% confidence")

    st.subheader("All class probabilities")
    for i, p in enumerate(probs):
        label = f"Digit {i}  {'✅' if i == digit else ''}"
        st.progress(float(p), text=f"{label}: {p*100:.1f}%")
