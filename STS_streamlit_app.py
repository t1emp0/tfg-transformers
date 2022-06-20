import streamlit as st

import numpy as np
import pandas as pd

if st.checkbox("Use multi-QA predictions"):
    pairs = pd.read_pickle("samples/pairs_sentences_QA.pkl")
else:
    pairs = pd.read_pickle("samples/pairs_sentences.pkl")

df = pd.read_pickle("samples/df_sentences.pkl")
sentences = (
    pd.concat([df["sentence1"], df["sentence2"]], axis=0)
    .reset_index(drop=True)
    .rename("Sentences")
)


i = st.slider("Selecciona la frase", max_value=len(sentences))

# Show original sentence, ground truth and predicted sentences
st.markdown(f"##### {sentences.iloc[i]}")
ids, scores = pairs.iloc[i][1], pairs.iloc[i][2]

i_norm = i - 200 if i >= 200 else i
i_pair = i - 200 if i >= 200 else i + 200
pair_found = i_pair in ids

st.markdown(
    f"\
  GT: {df.iloc[i_norm]['avg']} &emsp;&emsp; \
  Pair: {i_pair} &emsp;&emsp; \
  Found: {pair_found} &emsp;&emsp; \
  First: {(np.abs(ids[0] - i)== 200)} \
  "
)

if not pair_found:
    st.markdown(
        f"\
    **Matching sentence:** {sentences[i_pair]} \
    "
    )

st.table(
    pd.DataFrame(scores, ids).rename(columns={0: "score"}).join(sentences.iloc[ids])
)
