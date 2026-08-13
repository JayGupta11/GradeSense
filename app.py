import tempfile
import os
import streamlit as st

from blur_detection import check_image_quality
from pipeline import evaluate_answer_sheet

st.set_page_config(page_title="Handwritten Answer Evaluator", page_icon="📝", layout="centered")

st.title("📝 Handwritten Answer Evaluation (OCR + NLP)")
st.caption(
    "Upload a photo/scan of a handwritten answer. Blurry images are "
    "automatically rejected so you can retake the photo before it's scored."
)

with st.expander("1. Question setup", expanded=True):
    model_answer = st.text_area(
        "Model answer",
        placeholder="Enter the correct/reference answer here...",
        height=140,
    )
    keywords_raw = st.text_input(
        "Important keywords (comma-separated, optional — auto-extracted if left blank)"
    )
    max_marks = st.number_input("Maximum marks", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    blur_threshold = st.slider(
        "Blur sensitivity (lower = stricter, rejects more images)",
        min_value=20.0, max_value=300.0, value=100.0, step=5.0,
    )

st.divider()
st.subheader("2. Upload the student's answer image")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to a temp file so OpenCV/EasyOCR can read it by path
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded image", use_container_width=True)

    quality = check_image_quality(tmp_path, threshold=blur_threshold)

    if not quality["accepted"]:
        # BLOCK the upload
        st.error(
            f"🚫 Image rejected — too blurry (sharpness score: "
            f"{quality['blur_score']}, minimum required: {blur_threshold})."
        )
        st.warning(
            "Please retake the photo: use good lighting, hold the camera "
            "steady, and make sure the whole answer is in focus. Then "
            "upload it again above."
        )
    else:
        st.success(f"✅ Image quality OK (sharpness score: {quality['blur_score']}). Evaluating...")

        if not model_answer.strip():
            st.info("Enter a model answer above to run the evaluation.")
        else:
            keywords = (
                [k.strip() for k in keywords_raw.split(",") if k.strip()]
                if keywords_raw.strip()
                else None
            )

            with st.spinner("Running OCR and NLP evaluation..."):
                result = evaluate_answer_sheet(
                    image_path=tmp_path,
                    model_answer=model_answer,
                    keywords=keywords,
                    max_marks=max_marks,
                    blur_threshold=blur_threshold,
                )

            if result.status == "ERROR":
                st.error(result.message)
            else:
                st.subheader("3. Extracted answer (OCR)")
                st.code(result.ocr_text or "(empty)", language=None)
                st.caption(f"OCR engine: {result.ocr_engine}")

                ev = result.evaluation
                st.subheader("4. Score & Feedback")
                col1, col2 = st.columns(2)
                col1.metric("Marks", f"{ev['marks']} / {ev['max_marks']}")
                col2.metric("Similarity", f"{ev['similarity']*100:.1f}%")

                st.progress(min(ev["keyword_coverage"], 1.0), text=f"Keyword coverage: {ev['keyword_coverage']*100:.0f}%")

                if ev["matched_keywords"]:
                    st.write("✅ **Matched keywords:**", ", ".join(ev["matched_keywords"]))
                if ev["missing_keywords"]:
                    st.write("⚠️ **Missing keywords:**", ", ".join(ev["missing_keywords"]))

                st.write("**Feedback:**", ev["feedback"])