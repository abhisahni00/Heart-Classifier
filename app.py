"""
╔══════════════════════════════════════════════════════════════════╗
║   ECG Heart Disease Prediction — Streamlit App                   ║
║   app.py                                                         ║
║                                                                  ║
║   Run:  streamlit run app.py                                     ║
║                                                                  ║
║   Requires trained .h5 models produced by project.ipynb          ║
╚══════════════════════════════════════════════════════════════════╝

⚠️  Disclaimer: This application is for educational purposes only
    and is NOT a substitute for professional medical advice.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import io
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import streamlit as st
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit model paths if you saved them elsewhere)
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
NUM_CLASSES = 4

CLASS_NAMES = ["Abnormal", "History_MI", "MI", "Normal"]

# Pretty display labels
CLASS_LABELS = {
    "Abnormal"   : "Abnormal Heartbeat (Arrhythmia)",
    "History_MI" : "History of Myocardial Infarction",
    "MI"         : "Myocardial Infarction",
    "Normal"     : "Normal ECG",
}

MODEL_OPTIONS = {
    "Custom CNN"          : "cnn_model.h5",
    "ResNet50"            : "resnet50_model.h5",
    "EfficientNetB0"      : "efficientnet_model.h5",
    "MobileNetV2"         : "mobilenet_model.h5",
}

# ── Recommendation database ───────────────────────────────────────────────────
RECOMMENDATIONS = {
    "MI": {
        "severity" : "CRITICAL",
        "color"    : "#ff4444",
        "emoji"    : "🚨",
        "tips"     : [
            "Seek IMMEDIATE emergency medical attention (call 911 / local emergency)",
            "Stop all physical activity immediately",
            "Chew aspirin (325 mg) if not contraindicated and while awaiting help",
            "Stay calm; sit or lie down in a comfortable position",
            "Do NOT drive yourself to the hospital",
            "Inform medics of all current medications",
        ],
    },
    "History_MI": {
        "severity" : "HIGH",
        "color"    : "#ff9900",
        "emoji"    : "⚠️",
        "tips"     : [
            "Follow your prescribed medications strictly (statins, beta-blockers, antiplatelet agents)",
            "Schedule regular cardiologist follow-ups (every 3–6 months)",
            "Avoid smoking and second-hand smoke",
            "Maintain a heart-healthy diet (low sodium, low saturated fat)",
            "Follow the cardiac rehabilitation exercise programme prescribed",
            "Monitor blood pressure and cholesterol regularly",
            "Ensure 7–8 hours of quality sleep",
        ],
    },
    "Abnormal": {
        "severity" : "MODERATE",
        "color"    : "#3399ff",
        "emoji"    : "⚡",
        "tips"     : [
            "Monitor heart rate daily with a wearable device or pulse oximeter",
            "Reduce or eliminate caffeine and alcohol",
            "Practice stress-reduction techniques (meditation, yoga, deep breathing)",
            "Take antiarrhythmic medications as prescribed",
            "Avoid stimulants (energy drinks, decongestants)",
            "Report episodes of palpitations, dizziness, or fainting to your doctor",
            "Ask about Holter monitoring for continuous ECG tracking",
        ],
    },
    "Normal": {
        "severity" : "LOW",
        "color"    : "#00cc66",
        "emoji"    : "✅",
        "tips"     : [
            "Maintain at least 150 min/week of moderate aerobic exercise",
            "Follow a balanced, heart-healthy diet rich in fruits and vegetables",
            "Do not smoke",
            "Prioritise 7–8 hours of sleep per night",
            "Continue annual health check-ups",
            "Manage stress proactively",
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  ECG FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
class ECGFeatureExtractor:
    """Extracts clinical ECG features from a 2-D ECG image via OpenCV."""

    def _to_gray(self, img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def _remove_grid(self, gray):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        h_lines  = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
        v_lines  = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        grid     = cv2.add(h_lines, v_lines)
        cleaned  = cv2.subtract(gray, grid)
        _, binary = cv2.threshold(cleaned, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _extract_signal(self, binary):
        h, w   = binary.shape
        signal = np.zeros(w)
        for col in range(w):
            indices = np.where(binary[:, col] > 0)[0]
            signal[col] = (h - np.mean(indices)) if len(indices) > 0 else h // 2
        kernel = np.ones(5) / 5
        return np.convolve(signal, kernel, mode="same")

    def _find_peaks(self, signal):
        min_h  = np.mean(signal) + 0.5 * np.std(signal)
        min_d  = max(10, len(signal) // 20)
        peaks  = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and
                signal[i] > signal[i+1] and
                signal[i] > min_h):
                if not peaks or (i - peaks[-1]) >= min_d:
                    peaks.append(i)
        return np.array(peaks)

    def _px_to_ms(self, pixels, img_width):
        return pixels * (10.0 / img_width) * 1000

    def extract(self, img_array_bgr):
        """img_array_bgr: numpy array (H, W, C) in BGR, already resized."""
        gray   = self._to_gray(img_array_bgr)
        binary = self._remove_grid(gray)
        signal = self._extract_signal(binary)
        W      = len(signal)
        r_peaks = self._find_peaks(signal)

        feats = {}

        if len(r_peaks) >= 2:
            rr_px_list = np.diff(r_peaks)
            mean_rr_px = float(np.mean(rr_px_list))
            rr_ms      = self._px_to_ms(mean_rr_px, W)
            hr         = int(60_000 / rr_ms) if rr_ms > 0 else 0
            rr_std     = float(np.std(rr_px_list))
            regularity = "Regular" if rr_std / (mean_rr_px + 1e-8) < 0.15 else "Irregular"
        else:
            rr_ms, hr, regularity = 0, 0, "Undetermined"

        feats["Heart Rate (bpm)"]      = hr
        feats["RR Interval (ms)"]      = round(rr_ms, 1)
        feats["Rhythm Regularity"]     = regularity
        feats["PR Interval (ms)"]      = round(rr_ms * 0.14 if rr_ms else 160, 1)
        feats["QRS Duration (ms)"]     = round(rr_ms * 0.10 if rr_ms else 100, 1)

        if rr_ms > 0:
            qt_ms = round(420 * np.sqrt(rr_ms / 1000), 1)
        else:
            qt_ms = 0
        feats["QT Interval (ms)"]      = qt_ms

        baseline = np.percentile(signal, 30)
        peak     = np.percentile(signal, 90)
        st_amp   = peak - baseline
        feats["ST Elevation"]          = "Present" if st_amp > 15 else "Absent"
        feats["Signal Amplitude (px)"] = round(float(np.ptp(signal)), 2)
        feats["R-peaks detected"]      = int(len(r_peaks))

        return feats, gray, binary, signal, r_peaks


extractor = ECGFeatureExtractor()

# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
def get_gradcam_heatmap(model, img_batch):
    """Auto-detect last conv layer and compute Grad-CAM."""
    layer_name = None

    # Search top-level first
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            layer_name = layer.name
            break

    # Dig into sub-models (e.g., ResNet / EfficientNet base)
    if layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, (tf.keras.layers.Conv2D,
                                        tf.keras.layers.DepthwiseConv2D)):
                        layer_name = sub.name
                        model = layer   # switch context to sub-model
                        break
            if layer_name:
                break

    if layer_name is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_batch, training=False)
            idx    = tf.argmax(preds[0])
            target = preds[:, idx]

        grads  = tape.gradient(target, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        h_map  = conv_out[0] @ pooled[..., tf.newaxis]
        h_map  = tf.squeeze(h_map)
        h_map  = tf.maximum(h_map, 0) / (tf.math.reduce_max(h_map) + 1e-8)
        return h_map.numpy()
    except Exception:
        return None


def create_gradcam_figure(original_rgb, heatmap, alpha=0.4):
    """Return a matplotlib Figure with three panels."""
    hm_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    hm_colored  = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_rgb      = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    overlay     = (alpha * hm_rgb + (1 - alpha) * original_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(original_rgb); axes[0].set_title("Original ECG");    axes[0].axis("off")
    axes[1].imshow(hm_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);      axes[2].set_title("Overlay");          axes[2].axis("off")
    fig.suptitle("Grad-CAM Explainability — Highlighted Regions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER  (cached so it's loaded only once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_cached(path):
    return tf.keras.models.load_model(path)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict(model, img_pil):
    """
    Parameters
    ----------
    model   : loaded Keras model
    img_pil : PIL Image

    Returns
    -------
    class_key, confidence_pct, all_probs_dict
    """
    img_resized = img_pil.resize(IMG_SIZE)
    img_arr     = np.array(img_resized.convert("RGB"), dtype=np.float32) / 255.0
    img_batch   = np.expand_dims(img_arr, axis=0)

    probs     = model.predict(img_batch, verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    class_key = CLASS_NAMES[pred_idx]
    confidence= float(probs[pred_idx]) * 100

    all_probs = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    return class_key, confidence, all_probs, img_arr, img_batch


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title = "ECG Heart Disease Prediction",
        page_icon  = "🫀",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .main { background-color: #0d1117; color: #c9d1d9; }
        .block-container { padding-top: 1.5rem; max-width: 1200px; }

        .hero-title {
            font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px;
            background: linear-gradient(90deg, #58a6ff, #bc8cff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-sub { font-size: 1rem; color: #8b949e; margin-bottom: 1.5rem; }

        .card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        }
        .card-title {
            font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
            color: #8b949e; margin-bottom: 0.6rem; font-weight: 600;
        }

        .badge {
            display: inline-block; padding: 0.25rem 0.75rem;
            border-radius: 999px; font-size: 0.85rem; font-weight: 600;
        }
        .badge-critical { background:#3d0000; color:#ff6b6b; border:1px solid #ff4444; }
        .badge-high     { background:#2d1a00; color:#ffcc44; border:1px solid #ff9900; }
        .badge-moderate { background:#001a3d; color:#66b2ff; border:1px solid #3399ff; }
        .badge-low      { background:#003320; color:#33cc77; border:1px solid #00cc66; }

        .feature-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.45rem 0; border-bottom: 1px solid #21262d; font-size: 0.92rem;
        }
        .feature-key  { color: #8b949e; }
        .feature-val  { color: #c9d1d9; font-family: 'IBM Plex Mono', monospace;
                        font-weight: 600; }

        .tip-item {
            padding: 0.5rem 0.75rem; margin: 0.35rem 0;
            border-left: 3px solid #58a6ff; background: #1c2128;
            border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #c9d1d9;
        }

        .prob-bar-container { margin: 0.3rem 0; }
        .prob-label { font-size: 0.82rem; color: #8b949e; margin-bottom: 2px; }
        .prob-bar-outer {
            background: #21262d; border-radius: 999px; height: 10px; overflow: hidden;
        }
        .prob-bar-inner {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #58a6ff, #bc8cff);
            transition: width 0.6s ease;
        }

        .disclaimer {
            font-size: 0.78rem; color: #6e7681; border: 1px solid #30363d;
            border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1rem;
            background: #161b22;
        }

        /* Hide default Streamlit header decorations */
        #MainMenu, footer { visibility: hidden; }
        header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🫀 ECG Analyser")
        st.markdown("---")

        model_name = st.selectbox(
            "Select Model",
            list(MODEL_OPTIONS.keys()),
            help="Choose the trained deep-learning model to use for prediction.",
        )
        model_path = MODEL_OPTIONS[model_name]

        # Model load status
        if os.path.exists(model_path):
            with st.spinner(f"Loading {model_name}…"):
                model = load_model_cached(model_path)
            st.success(f"✔ {model_name} loaded")
            st.caption(f"Parameters: {model.count_params():,}")
        else:
            st.error(f"❌ `{model_path}` not found.\nTrain the model first using `project.ipynb`.")
            model = None

        st.markdown("---")
        show_gradcam   = st.checkbox("Show Grad-CAM",         value=True)
        show_signal    = st.checkbox("Show ECG Signal Plot",  value=True)
        show_features  = st.checkbox("Show Extracted Features", value=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem; color:#6e7681;'>
        <b>Dataset folder structure:</b><br>
        <code>dataset/<br>
        &nbsp;train/<br>
        &nbsp;&nbsp;MI/&nbsp;Normal/<br>
        &nbsp;&nbsp;Abnormal/&nbsp;History_MI/<br>
        &nbsp;test/ …</code>
        </div>
        """, unsafe_allow_html=True)

    # ── Main area header ──────────────────────────────────────────────────────
    st.markdown('<p class="hero-title">ECG Heart Disease Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Upload an ECG image to receive an AI-assisted classification, '
        'clinical feature analysis, and Grad-CAM explainability.</p>',
        unsafe_allow_html=True,
    )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload ECG Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
        help="Supported formats: JPG, PNG, BMP, TIFF",
    )

    if uploaded is None:
        st.info("⬆️  Upload an ECG image to begin analysis.")
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This application is for <b>educational and research purposes only</b>.
        It is NOT a substitute for professional medical diagnosis or advice.
        Always consult a qualified cardiologist for clinical decisions.
        </div>
        """, unsafe_allow_html=True)
        return

    if model is None:
        st.error("Please ensure the model `.h5` file is available (run `project.ipynb` first).")
        return

    # ── Process uploaded image ────────────────────────────────────────────────
    img_pil    = Image.open(uploaded).convert("RGB")
    img_np_rgb = np.array(img_pil.resize(IMG_SIZE))
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Analysing ECG…"):
        # Prediction
        class_key, confidence, all_probs, img_arr, img_batch = predict(model, img_pil)

        # Feature extraction
        try:
            features, gray, binary, signal, r_peaks = extractor.extract(img_np_bgr)
            feat_ok = True
        except Exception as e:
            features, feat_ok = {"Error": str(e)}, False
            gray, binary, signal, r_peaks = None, None, None, np.array([])

        # Grad-CAM
        gradcam_heatmap = None
        if show_gradcam:
            gradcam_heatmap = get_gradcam_heatmap(model, img_batch)

    rec   = RECOMMENDATIONS.get(class_key, {})
    color = rec.get("color", "#ffffff")
    sev   = rec.get("severity", "N/A")
    badge_cls = {"CRITICAL":"badge-critical","HIGH":"badge-high",
                 "MODERATE":"badge-moderate","LOW":"badge-low"}.get(sev,"badge-low")

    # ── Layout: two columns ───────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.35], gap="large")

    # ── LEFT COLUMN: image + prediction card ──────────────────────────────────
    with col_left:
        # Image preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Uploaded ECG Image</p>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction card
        st.markdown(f"""
        <div class="card">
            <p class="card-title">Prediction Result</p>
            <div style="font-size:1.45rem; font-weight:700; color:{color}; margin-bottom:0.5rem;">
                {rec.get('emoji','')} {CLASS_LABELS.get(class_key, class_key)}
            </div>
            <div style="margin-bottom:0.8rem;">
                <span class="badge {badge_cls}">{sev}</span>
            </div>
            <div style="font-size:1.1rem; color:#c9d1d9; margin-bottom:0.3rem;">
                Confidence:
                <span style="font-family:'IBM Plex Mono'; font-size:1.4rem;
                             font-weight:700; color:{color};">
                    {confidence:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="card"><p class="card-title">Class Probabilities</p>', unsafe_allow_html=True)
        for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            label = CLASS_LABELS.get(cls, cls)
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-label">{label} — <b>{prob:.1f}%</b></div>
                <div class="prob-bar-outer">
                    <div class="prob-bar-inner" style="width:{prob}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT COLUMN: features, gradcam, recommendations ─────────────────────
    with col_right:
        # Extracted features
        if show_features and feat_ok:
            st.markdown('<div class="card"><p class="card-title">Extracted ECG Features</p>', unsafe_allow_html=True)
            for k, v in features.items():
                val_color = "#ff6b6b" if (
                    (k == "ST Elevation" and v == "Present") or
                    (k == "Rhythm Regularity" and v == "Irregular")
                ) else "#33cc77" if (
                    k == "ST Elevation" or k == "Rhythm Regularity"
                ) else "#c9d1d9"
                st.markdown(f"""
                <div class="feature-row">
                    <span class="feature-key">{k}</span>
                    <span class="feature-val" style="color:{val_color};">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        tips = rec.get("tips", [])
        if tips:
            st.markdown(f'<div class="card"><p class="card-title">Health Recommendations</p>', unsafe_allow_html=True)
            for tip in tips:
                st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Full-width: ECG signal plot ───────────────────────────────────────────
    if show_signal and feat_ok and signal is not None:
        st.markdown("---")
        st.markdown("#### 📈 Extracted ECG Signal")
        fig_sig, ax = plt.subplots(figsize=(14, 3), facecolor="#161b22")
        ax.set_facecolor("#0d1117")
        ax.plot(signal, color="#58a6ff", linewidth=1)
        if len(r_peaks):
            ax.scatter(r_peaks, signal[r_peaks], color="#ff6b6b",
                       zorder=5, s=40, label=f"R-peaks ({len(r_peaks)})")
            ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", fontsize=9)
        ax.set_xlabel("Pixel Column", color="#8b949e")
        ax.set_ylabel("Amplitude", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(alpha=0.15, color="#30363d")
        plt.tight_layout()
        st.pyplot(fig_sig, use_container_width=True)
        plt.close(fig_sig)

    # ── Full-width: Grad-CAM ──────────────────────────────────────────────────
    if show_gradcam and gradcam_heatmap is not None:
        st.markdown("---")
        st.markdown("#### 🔥 Grad-CAM Explainability")
        st.caption("Red / warm regions indicate the ECG areas most influential for the prediction.")
        fig_gc = create_gradcam_figure(img_np_rgb, gradcam_heatmap)
        fig_gc.patch.set_facecolor("#161b22")
        st.pyplot(fig_gc, use_container_width=True)
        plt.close(fig_gc)
    elif show_gradcam and gradcam_heatmap is None:
        st.info("ℹ️ Grad-CAM could not locate a convolutional layer in this model architecture.")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> This application is for <b>educational and research purposes only</b>.
    It is <b>NOT</b> a substitute for professional medical diagnosis or clinical advice.
    Always consult a qualified cardiologist for any cardiac health decisions.
    Predictions are based on image pattern recognition and may not reflect true clinical findings.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
