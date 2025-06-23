import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import requests # Added for manual download
import zipfile # Added for manual extraction
import io # Added for in-memory file handling

# Define a local path for NLTK data within the mounted app directory
# This path *must* be relative to the app's root on Streamlit Cloud
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Set NLTK data path as early as possible
# This tells NLTK where to look for data.
# It's crucial to do this *before* any NLTK function that requires data is called.
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
    st.info(f"Configuring NLTK data path: {NLTK_DATA_DIR}")

# Import clustering functions AFTER NLTK data path is set, but BEFORE NLTK functions are used
from clustering import cluster_articles, get_top_keywords, preprocess_text

# --- Robust NLTK Data Download and Path Setup ---
@st.cache_resource
def download_and_check_nltk_data():
    """
    Downloads necessary NLTK data for text preprocessing and verifies its presence.
    Uses direct download and extraction for problematic datasets.
    """
    # Create the base NLTK data directory if it doesn't exist
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR)
        st.info(f"Created NLTK data directory: {NLTK_DATA_DIR}")

    # Define common NLTK data URL
    NLTK_DOWNLOAD_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"

    # List of NLTK datasets to download and verify
    # Use direct download for wordnet and omw-1.4 due to past issues
    # Keep nltk.download for punkt and stopwords as they are usually fine
    datasets = {
        'punkt': {'path': os.path.join(NLTK_DATA_DIR, 'tokenizers', 'punkt'), 'type': 'nltk_download'},
        'stopwords': {'path': os.path.join(NLTK_DATA_DIR, 'corpora', 'stopwords'), 'type': 'nltk_download'},
        'wordnet': {'path': os.path.join(NLTK_DATA_DIR, 'corpora', 'wordnet'), 'type': 'manual', 'zip_file': 'wordnet.zip'},
        'omw-1.4': {'path': os.path.join(NLTK_DATA_DIR, 'corpora', 'omw-1.4'), 'type': 'manual', 'zip_file': 'omw-1.4.zip'}
    }

    all_downloads_successful = True
    for dataset, info in datasets.items():
        path_to_check = info['path']
        download_type = info['type']

        if not os.path.exists(path_to_check):
            st.info(f"NLTK '{dataset}' not found at '{path_to_check}'. Attempting download...")
            try:
                if download_type == 'nltk_download':
                    # Use force=True for nltk.download to ensure it tries again
                    nltk.download(dataset, download_dir=NLTK_DATA_DIR, quiet=True, force=True)
                    if os.path.exists(path_to_check):
                        st.success(f"NLTK '{dataset}' downloaded and verified.")
                    else:
                        st.error(f"NLTK '{dataset}' download completed, but data not found at expected path: {path_to_check}.")
                        all_downloads_successful = False
                elif download_type == 'manual':
                    zip_file_name = info['zip_file']
                    zip_url = NLTK_DOWNLOAD_URL + zip_file_name
                    
                    st.info(f"Manually downloading '{zip_file_name}' from {zip_url}...")
                    response = requests.get(zip_url, stream=True)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    # Extract directly from memory
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        # Extract all contents to the NLTK_DATA_DIR/corpora path
                        # The zip file itself contains 'wordnet' or 'omw-1.4' folder at its root
                        # So, extracting to NLTK_DATA_DIR/corpora will place it correctly.
                        target_dir_for_zip_content = os.path.join(NLTK_DATA_DIR, 'corpora')
                        os.makedirs(target_dir_for_zip_content, exist_ok=True) # Ensure corpora dir exists
                        zf.extractall(target_dir_for_zip_content)
                    
                    if os.path.exists(path_to_check):
                        st.success(f"NLTK '{dataset}' manually downloaded and extracted to {path_to_check}.")
                    else:
                        st.error(f"NLTK '{dataset}' manual download completed, but data not found at expected path: {path_to_check}. This may indicate an issue with extraction or file structure within the zip.")
                        all_downloads_successful = False

            except Exception as e:
                st.error(f"Failed to download or extract NLTK '{dataset}' data: {e}. App will stop.")
                all_downloads_successful = False
        else:
            st.info(f"NLTK '{dataset}' already present at '{path_to_check}'. Skipping download.")

    if all_downloads_successful:
        st.success("All required NLTK data is available.")
        return True
    else:
        st.error("One or more NLTK datasets failed to download or verify. Application cannot proceed.")
        return False

# Call the NLTK data download and check function at the very start of your app logic
if not download_and_check_nltk_data():
    st.stop() # Stop the Streamlit app gracefully if data is not available


# Set Streamlit page configuration
st.set_page_config(page_title="News Section Explorer", layout="wide") # Corrected st.set_page_config to st.set_config

# --- Data Loading Function (Cached) ---
@st.cache_data
def load_data(csv_path='news.csv'):
    """
    Loads news data from CSV, performs initial preprocessing, and clustering.
    Caches the results to improve app performance.
    """
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

    # Call cluster_articles from the clustering.py module
    df, section_names = cluster_articles(df) # cluster_articles now takes df directly

    st.success("Data loaded and clustered successfully!")
    return df, section_names

# --- Main Streamlit App Layout and Logic ---
def main_app():
    st.title("News Section Clustering and Keyword Explorer")

    # Load data (this function is cached)
    df, section_names = load_data()

    if df.empty:
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
