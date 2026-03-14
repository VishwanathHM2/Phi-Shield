"""
ui/streamlit_app.py

Streamlit UI for PHI De-identification System.

Modes:
  1. TEXT MODE   — Paste clinical text, detect + redact PHI
  2. REPORT MODE — Upload PDF, extract text, detect + redact PHI
"""

import sys
from pathlib import Path
import tempfile
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="PHI De-identification System",
    page_icon="🏥",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading PHI detection model...")
def load_predictor():
    try:
        from inference.predict import PHIPredictor
        from config.config import BEST_MODEL_PATH, VOCAB_FILE
        if not BEST_MODEL_PATH.exists() or not VOCAB_FILE.exists():
            return None
        return PHIPredictor()
    except Exception as e:
        st.warning(f"Model not loaded: {e}. Using regex-only mode.")
        return None


def get_redactor(predictor):
    from inference.redact_text import hybrid_redact, redact_document
    return hybrid_redact, redact_document


ENTITY_COLORS = {
    "NAME": "#FF6B6B",
    "DOCTOR": "#FF8E53",
    "HOSPITAL": "#4ECDC4",
    "LOCATION": "#45B7D1",
    "DATE": "#96CEB4",
    "PHONE": "#FFEAA7",
    "EMAIL": "#DDA0DD",
    "ID": "#98D8C8",
}


def highlight_entities(tokens, tags):
    html_parts = []
    i = 0

    while i < len(tags):
        tag = tags[i]

        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1

            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1

            span_text = " ".join(tokens[i:j])
            color = ENTITY_COLORS.get(etype, "#E0E0E0")

            html_parts.append(
                f'<mark style="background:{color};padding:2px 6px;border-radius:4px;'
                f'font-weight:bold;" title="{etype}">{span_text} '
                f'<sup style="font-size:0.7em">{etype}</sup></mark>'
            )

            i = j

        else:
            html_parts.append(tokens[i])
            i += 1

    return " ".join(html_parts)


def render_entity_table(entities):
    if not entities:
        st.info("No PHI entities detected.")
        return

    import pandas as pd

    rows = [{"Entity Type": e["entity_type"], "Value": e["text"]} for e in entities]

    df = pd.DataFrame(rows).drop_duplicates()

    st.dataframe(df, use_container_width=True)


def main():

    st.title("🏥 Clinical PHI De-identification System")

    st.caption("Detect and anonymize Protected Health Information from medical records.")

    predictor = load_predictor()

    if predictor is None:
        st.warning(
            "⚠️ Trained model not found. Running in regex-only mode. "
            "Train the model first (python training/train.py)."
        )

    hybrid_redact, redact_document = get_redactor(predictor)

    mode = st.sidebar.radio(
        "Select Mode",
        ["📝 Text Mode", "📄 Report Mode (PDF)"],
    )

    with st.sidebar.expander("Entity Color Legend"):
        for etype, color in ENTITY_COLORS.items():
            st.markdown(
                f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                f'font-weight:bold;">{etype}</span>',
                unsafe_allow_html=True,
            )

    if mode == "📝 Text Mode":

        st.header("Text Mode — Clinical Report De-identification")

        default_text = (
            "Patient Ravi Kumar visited Apollo Hospital in Bangalore on 12 March 2024. "
            "Dr Priya Sharma examined him and noted elevated blood pressure. "
            "Contact patient at 9876543210 or email ravi.kumar@gmail.com. "
            "Patient ID: PT-445521. Follow-up scheduled for 26 March 2024."
        )

        input_text = st.text_area(
            "Enter clinical text:",
            value=default_text,
            height=180,
            key="input_text"
        )

        if st.button("🔍 Detect & Redact PHI", type="primary"):

            if not input_text.strip():
                st.warning("Please enter some text.")
                return

            with st.spinner("Detecting PHI..."):

                result = redact_document(
                    input_text,
                    predictor,
                    use_model=(predictor is not None),
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Text")

                st.text_area(
                    "Original Text",
                    value=result["original"],
                    height=200,
                    disabled=True,
                    key="text_original_output"
                )

            with col2:

                st.subheader("Redacted Text")

                st.text_area(
                    "Redacted Text",
                    value=result["redacted"],
                    height=200,
                    disabled=True,
                    key="text_redacted_output"
                )

            st.subheader("Entity Highlighting")

            if predictor:

                try:

                    tokens, tags = predictor.predict_tags(input_text)

                    highlighted = highlight_entities(tokens, tags)

                    st.markdown(
                        f'<div style="background:#f8f9fa;padding:16px;border-radius:8px;'
                        f'line-height:2.0;font-size:1.05em">{highlighted}</div>',
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.info(f"Highlighting unavailable: {e}")

            else:
                st.info("Entity highlighting requires a trained model.")

            st.subheader("Detected PHI Entities")

            render_entity_table(result["entities"])

            st.download_button(
                label="⬇️ Download Redacted Text",
                data=result["redacted"],
                file_name="redacted_report.txt",
                mime="text/plain",
            )

    elif mode == "📄 Report Mode (PDF)":

        st.header("Report Mode — PDF De-identification")

        uploaded_file = st.file_uploader(
            "Upload a medical report (PDF)",
            type=["pdf"],
            key="pdf_upload"
        )

        if uploaded_file:

            with st.spinner("Extracting text from PDF..."):

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:

                    tmp.write(uploaded_file.read())

                    tmp_path = Path(tmp.name)

                try:

                    from utils.pdf_parser import extract_text_from_pdf

                    raw_text = extract_text_from_pdf(tmp_path)

                finally:

                    tmp_path.unlink(missing_ok=True)

            if not raw_text.strip():

                st.error("Could not extract text from PDF.")

                return

            st.success(f"Extracted {len(raw_text):,} characters from PDF.")

            with st.expander("View Extracted Text"):

                st.text_area(
                    "Raw extracted text",
                    value=raw_text,
                    height=300,
                    disabled=True,
                    key="pdf_raw_text"
                )

            if st.button("🔍 Run PHI Detection", type="primary"):

                with st.spinner("Detecting and redacting PHI..."):

                    result = redact_document(
                        raw_text,
                        predictor,
                        use_model=(predictor is not None),
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Original Text")

                    st.text_area(
                        "Original Text Preview",
                        value=result["original"][:3000] + "...",
                        height=400,
                        disabled=True,
                        key="pdf_original_preview"
                    )

                with col2:

                    st.subheader("Redacted Text")

                    st.text_area(
                        "Redacted Text Preview",
                        value=result["redacted"][:3000] + "...",
                        height=400,
                        disabled=True,
                        key="pdf_redacted_preview"
                    )

                st.subheader("Detected PHI Entities")

                render_entity_table(result["entities"])

                col_a, col_b = st.columns(2)

                with col_a:

                    st.download_button(
                        label="⬇️ Download Redacted Text",
                        data=result["redacted"],
                        file_name=f"redacted_{uploaded_file.name.replace('.pdf', '.txt')}",
                        mime="text/plain",
                    )

                with col_b:

                    import json

                    entity_json = json.dumps(result["entities"], indent=2)

                    st.download_button(
                        label="⬇️ Download Entity Report (JSON)",
                        data=entity_json,
                        file_name="phi_entities.json",
                        mime="application/json",
                    )


if __name__ == "__main__":
    main()