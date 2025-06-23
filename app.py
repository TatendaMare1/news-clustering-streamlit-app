import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# CORRECTED IMPORT from clustering.py:
from clustering import cluster_articles, get_top_keywords, preprocess_text

# --- Robust NLTK Data Download and Path Setup ---
# Define a local path for NLTK data within the mounted app directory
# This path *must* be relative to the app's root on Streamlit Cloud
# os.path.dirname(__file__) gives the directory of the current script (app.py)
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Set NLTK data path as early as possible
# This tells NLTK where to look for data.
# It's crucial to do this *before* any NLTK function that requires data is called.
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
    st.info(f"Configuring NLTK data path: {NLTK_DATA_DIR}")

# Use st.cache_resource to download NLTK data only once per app deployment.
@st.cache_resource
def download_and_check_nltk_data():
    """
    Downloads necessary NLTK data for text preprocessing and verifies its presence.
    Forces download to a specific, writable directory within the app's scope.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR)
        st.info(f"Created NLTK data directory: {NLTK_DATA_DIR}")

    # Check if 'punkt' tokenizer data exists before attempting download
    # This helps avoid redundant downloads and verifies success
    punkt_path = os.path.join(NLTK_DATA_DIR, 'tokenizers', 'punkt')
    if not os.path.exists(punkt_path):
        try:
            st.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
            st.info("NLTK 'punkt' downloaded.")
        except Exception as e:
            st.error(f"Failed to download NLTK 'punkt' data: {e}. App will stop.")
            return False # Indicate failure

    # Check for other necessary datasets similarly
    # stopwords
    stopwords_path = os.path.join(NLTK_DATA_DIR, 'corpora', 'stopwords')
    if not os.path.exists(stopwords_path):
        try:
            st.info("NLTK 'stopwords' not found. Attempting download...")
            nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
            st.info("NLTK 'stopwords' downloaded.")
        except Exception as e:
            st.error(f"Failed to download NLTK 'stopwords' data: {e}. App will stop.")
            return False

    # wordnet
    wordnet_path = os.path.join(NLTK_DATA_DIR, 'corpora', 'wordnet')
    if not os.path.exists(wordnet_path):
        try:
            st.info("NLTK 'wordnet' not found. Attempting download...")
            nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True)
            st.info("NLTK 'wordnet' downloaded.")
        except Exception as e:
            st.error(f"Failed to download NLTK 'wordnet' data: {e}. App will stop.")
            return False

    # omw-1.4 (often required by WordNetLemmatizer)
    omw_path = os.path.join(NLTK_DATA_DIR, 'corpora', 'omw-1.4')
    if not os.path.exists(omw_path):
        try:
            st.info("NLTK 'omw-1.4' not found. Attempting download...")
            nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR, quiet=True)
            st.info("NLTK 'omw-1.4' downloaded.")
        except Exception as e:
            st.error(f"Failed to download NLTK 'omw-1.4' data: {e}. App will stop.")
            return False

    st.success("All required NLTK data is available.")
    return True # Indicate success

# Call the NLTK data download and check function at the very start of your app logic
if not download_and_check_nltk_data():
    st.error("Application cannot proceed due to NLTK data issues.")
    st.stop() # Stop the Streamlit app gracefully if data is not available


# Set Streamlit page configuration
st.set_page_config(page_title="News Section Explorer", layout="wide")

# --- Data Loading Function (Cached) ---
@st.cache_data
def load_data(csv_path='news.csv'):
    """
    Loads news data from CSV, performs initial preprocessing, and clustering.
    Caches the results to improve app performance.
    """
    # Ensure news.csv is in the correct path relative to app.py
    # If news.csv is not in the same directory as app.py, you'll need to adjust csv_path
    # Example: If news.csv is in a 'data' folder, use 'data/news.csv'
    if not os.path.exists(csv_path):
        st.error(f"Error: The file '{csv_path}' was not found. "
                 "Please ensure 'news.csv' is in the same directory as 'app.py' on GitHub, or update the path.")
        st.stop() # Stop the app if crucial data file is missing
        return pd.DataFrame(), []

    st.info(f"Loading data from {csv_path} and performing clustering. This might take a moment...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error reading CSV file '{csv_path}': {e}. Please check file format and content.")
        st.stop()
        return pd.DataFrame(), []

    # Perform clustering (which includes preprocessing text)
    df, section_names = cluster_articles(df) # cluster_articles now takes df directly

    st.success("Data loaded and clustered successfully!")
    return df, section_names

# --- Main Streamlit App Layout and Logic ---
def main_app():
    st.title("News Section Clustering and Keyword Explorer")

    # Load data (this function is cached)
    df, section_names = load_data()

    if df.empty:
        # load_data handles errors and stops, but this is a fallback for safety
        st.warning("No data to display. Application stopped or data could not be loaded.")
        return

    # Sidebar for filtering/selection
    st.sidebar.header("Filter and Explore")

    # Display basic info
    st.sidebar.write(f"Total articles loaded: {len(df)}")
    st.sidebar.write(f"Number of sections: {len(section_names)}")

    # --- Section Filtering ---
    selected_section = st.sidebar.selectbox(
        "Select a News Section:",
        ["All Sections"] + sorted(section_names.tolist())
    )

    filtered_df = df
    if selected_section != "All Sections":
        filtered_df = df[df['section'] == selected_section]
        st.subheader(f"Articles in '{selected_section}' Section ({len(filtered_df)} articles)")
    else:
        st.subheader(f"All News Articles ({len(filtered_df)} articles)")

    # Display filtered articles
    if not filtered_df.empty:
        st.dataframe(filtered_df[['title', 'text', 'section', 'cluster_name']].head(10))
        if st.checkbox("Show all filtered articles"):
            st.dataframe(filtered_df[['title', 'text', 'section', 'cluster_name']])
    else:
        st.info("No articles found for the selected filter.")


    # --- Display Top Keywords per Section (if clustering results are available) ---
    st.subheader("Top Keywords for Each News Section")

    if 'processed_text' in df.columns and 'cluster_name' in df.columns:
        top_keywords_dict = get_top_keywords(df, n=10)

        if top_keywords_dict:
            for section, keywords in top_keywords_dict.items():
                st.write(f"**Section: {section}**")
                keyword_str = ", ".join([f"{word} ({count})" for word, count in keywords])
                st.write(keyword_str)
        else:
            st.warning("Could not generate top keywords. Check 'processed_text' and 'cluster_name' columns.")
    else:
        st.warning("Processed text or cluster names not found in data for keyword analysis.")

    st.markdown("---")
    st.write("Developed by Tatenda Mare for news clustering and exploration.")


# --- Run the main Streamlit app ---
if __name__ == "__main__":
    main_app()
