import numpy as np
import streamlit as st
from PIL import Image

from heuristic import (
    crop_to_document_edges,
    convert_debug_to_summary,
    draw_overlays,
    evaluate_quality
)

# ---------------- Streamlit Page Config ----------------
st.set_page_config("Passport/ID Quality Checker", layout="centered")
st.title("🛂 Passport/ID Quality Checker")


# ---------------- Helper Functions ----------------
#
# def show_edge_examples():
#     st.markdown("### 📏 How to Capture Your Document Properly")
#     st.markdown("""
# To ensure the system detects the document edges correctly, please follow the examples below.
# Make sure the **entire document is visible**, including all **four edges and corners**.
#     """)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.markdown("#### ✅ Good Example")
#         st.image([
#             "C:/Users/ASUS/ImageQualityDetection/image/good.png"  # Diagram with cropped edges
#         ],
#             caption="Document fully visible with all edges exposed.",
#             use_container_width=True)
#
#     with col2:
#         st.markdown("#### ❌ Bad Example")
#         st.image([
#             "C:/Users/ASUS/ImageQualityDetection/image/bad.png"  # Diagram with cropped edges
#         ],
#             caption="Edges cut off — system may fail to detect layout.",
#             use_container_width=True)
#
#     st.markdown("""
# ### 📸 Tips for Good Edge Visibility
# - Place the document **flat** on a contrasting background (table, dark surface).
# - Keep **all borders visible**. Do not zoom too much.
# - Avoid fingers covering corners.
# - Make sure the background is not too cluttered.
# - Hold camera parallel (avoid tilted angles).
# """)
#
#
# show_edge_examples()

# ---------------- Streamlit UI ----------------
uploaded_file = st.file_uploader("Upload passport or ID image", type=["jpg", "jpeg", "png"])
QUALITY_THRESHOLD = st.slider("Quality Score Threshold", 50, 100, 70)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image quality..."):
        # Try cropping
        cropped_img, crop_box = crop_to_document_edges(image_np)

        # Evaluate either cropped or original
        if crop_box and cropped_img.shape != image_np.shape:
            score, reasons, debug, debug_image = evaluate_quality(cropped_img)
            image_to_show = debug_image
        else:
            score, reasons, debug, debug_image = evaluate_quality(image_np)
            image_to_show = debug_image

        # 🚫 EARLY STOP: If no document → DO NOT continue processing
        if not debug.get("document_like", False):
            st.error("❌ FAIL: No valid document structure detected.")
            st.markdown("### 🧾 Quality Score Summary")
            for line in reasons:
                st.markdown(f"- {line}")

            with st.expander("🔧 Internal Metrics"):
                st.json(convert_debug_to_summary(debug))
            st.stop()  # ← VERY IMPORTANT

        # If document is valid → continue normal workflow

        annotated_image = draw_overlays(debug_image, debug)

    # ---------------- Results Display ----------------
    st.image(annotated_image, caption="🖍️ Visual Feedback", use_container_width=True)

    st.markdown("### 🧾 Quality Score Summary")
    for line in reasons:
        st.markdown(f"- {line}")

    st.markdown(f"### 🎯 Final Score: **{score} / 100**")
    st.progress(min(score, 100) / 100)

    if score >= QUALITY_THRESHOLD:
        st.success("✅ PASS: Document meets quality standards.")
    else:
        st.error("❌ FAIL: Document doesn’t meet quality standards.")

    with st.expander("🔧 Internal Metrics"):
        st.json(convert_debug_to_summary(debug))

else:
    st.info("📤 Please upload a passport or ID image to begin.")



