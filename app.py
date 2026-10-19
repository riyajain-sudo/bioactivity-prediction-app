from pathlib import Path
import base64
import pickle

import pandas as pd
import streamlit as st
from PIL import Image

from prediction import (MAX_MOLECULES, MAX_UPLOAD_BYTES, PredictionInputError,
                       calculate_descriptors, parse_input_text)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "acetylcholinesterase_model.pkl"


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'


def predict_descriptors(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed local tutorial model to already validated descriptors."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model not found. Generate acetylcholinesterase_model.pkl by running the tutorial notebook.")
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    prediction = model.predict(descriptors.drop(columns=["molecule_name"]))
    return pd.DataFrame({"molecule_name": descriptors["molecule_name"].to_list(),
                         "pIC50": prediction})


image_path = APP_DIR / "logo.png"
st.image(Image.open(image_path), use_container_width=True)
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This tutorial fork predicts a computational pIC50 estimate for acetylcholinesterase.
It is for research and education only, not clinical advice or evidence of drug efficacy.

**Credits**
- App built in `Python` + `Streamlit` by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descriptor calculation using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
- This repository is a tutorial fork; retain attribution to the upstream project.
---
""")

st.sidebar.header("1. Upload whitespace-delimited input")
st.sidebar.caption("Each row: SMILES then a unique molecule ID (as in the example file).")
uploaded_file = st.sidebar.file_uploader("Upload input file", type=["txt"],
                                          help=f"Up to {MAX_MOLECULES} molecules and {MAX_UPLOAD_BYTES} bytes.")
st.sidebar.markdown("[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)")

if st.sidebar.button("Predict", disabled=uploaded_file is None):
    try:
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            raise PredictionInputError(f"Upload exceeds the {MAX_UPLOAD_BYTES}-byte limit.")
        text = uploaded_file.getvalue().decode("utf-8")
        rows = parse_input_text(text)
        st.header("Original input data")
        st.dataframe(rows, use_container_width=True)
        with st.spinner("Calculating descriptors in an isolated request workspace..."):
            descriptors = calculate_descriptors(rows, APP_DIR)
        st.header("Calculated molecular descriptors")
        st.write(descriptors.shape)
        predictions = predict_descriptors(descriptors)
        st.header("Prediction output")
        st.dataframe(predictions, use_container_width=True)
        st.markdown(filedownload(predictions), unsafe_allow_html=True)
    except (UnicodeDecodeError, PredictionInputError, FileNotFoundError) as exc:
        st.error(str(exc))
    except Exception:
        # Do not expose local paths, process output, or serialized model details to the browser.
        st.error("Prediction failed. Check the local PaDEL installation, runtime requirements, and generated model.")
else:
    st.info("Upload a valid text file in the sidebar to start.")
