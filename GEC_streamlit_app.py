import streamlit as st

import numpy as np
import pandas as pd


def _update_slider(value):
    st.session_state["article_slider"] = value
    st.session_state["article_box"] = value


pairs = pd.DataFrame(pd.read_pickle("samples/2_3_article_pairs_6k.pkl"))
df = pd.read_csv("samples/2_1_article_titles_6k.tsv", sep="\t")

cols_header = st.columns(4)

with cols_header[0]:
    show_suggestions = st.checkbox("Show suggestions")
with cols_header[1]:
    hide_preview = st.checkbox("Hide preview")
with cols_header[3]:
    st.button(
        "Random article", on_click=_update_slider, args=[np.random.randint(df.shape[0])]
    )


if show_suggestions:
    NUM_COLS = 4

    random_titles = df["title"].sample(NUM_COLS)

    cols = st.columns(NUM_COLS)
    for (i, col) in enumerate(cols):
        with col:
            st.write(random_titles.iloc[i])
            st.button(
                str(random_titles.index[i]),
                on_click=_update_slider,
                args=[int(random_titles.index[i])],
            )


# current = st.slider("Selecciona l'article", key="article_slider", max_value=df.shape[0])
current = st.selectbox(
    "Selecciona",
    list(range(df.shape[0])),
    key="article_box",
    format_func=lambda x: df["title"][x],
)

# Show original article, and predicted sentences
st.markdown(f"##### {df['title'].iloc[current]}")
st.write(df["preview"].iloc[current] + "...")

ids, scores = pairs.iloc[current][1], pairs.iloc[current][2]
current_df = pd.DataFrame(scores, ids).rename(columns={0: "scores"}).join(df.iloc[ids])

if hide_preview:
    st.table(current_df.drop(columns=["preview"]))
else:
    st.table(current_df)
