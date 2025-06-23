import streamlit as st
import pandas as pd
import os
# CORRECTED LINE:
from clustering import cluster_articles, get_top_keywords, preprocess_text

# Set page config
st.set_page_config(page_title="News Section Explorer", layout="wide")

# Load data
@st.cache_data
def load_data():
    df, section_names = cluster_articles()
    return df, section_names

df, section_names = load_data()
keywords_by_section = get_top_keywords(df)

# Main app
st.title("News Article Sections Explorer")

# Section selection
selected_section = st.selectbox(
    "Select a news section to explore",
    sorted(section_names)
)

# Display section content
section_df = df[df['cluster_name'] == selected_section]

st.header(f"Section: {selected_section}")

# Show most common words
st.subheader("Top Keywords in this Section")
keywords = keywords_by_section[selected_section]

# Display keywords in columns
cols = st.columns(5)
for i, (word, count) in enumerate(keywords):
    cols[i%5].metric(label=word, value=count)

# Show articles
st.subheader(f"Articles in {selected_section} Section")
for _, row in section_df.iterrows():
    with st.expander(row['title']):
        st.write(f"**Source:** {row['newspaper']}")
        st.write(f"**Date:** {row['date']}")
        st.write(row['text'][:500] + "...")
        st.write(f"[Read Full Article]({row['url']})")
