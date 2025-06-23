import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# CORRECTED IMPORT from clustering.py:
from clustering import cluster_articles, get_top_keywords, preprocess_text

# --- NLTK Data Download (Cached and Managed) ---
# Use st.cache_resource to download NLTK data only once per app deployment.
# This ensures the data is available for preprocessing functions.
@st.cache_resource
def download_nltk_data():
    """
    Downloads necessary NLTK data for text preprocessing.
    Forces download to a specific, writable directory within the app's scope.
    """
    # Define a local path for NLTK data within the mounted app directory
    # os.path.dirname(__file__) gives the directory of the current script (app.py)
    nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    # Add this custom path to NLTK's data search paths.
    # This tells NLTK where to look for data.
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
        st.info(f"Configuring NLTK data path: {nltk_data_path}")

    try:
        st.info("Attempting to download NLTK 'stopwords'...")
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        st.info("Attempting to download NLTK 'wordnet'...")
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        st.info("Attempting to download NLTK 'punkt'...")
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        # 'omw-1.4' is often needed by WordNetLemmatizer, so it's good to include
        st.info("Attempting to download NLTK 'omw-1.4' (for WordNetLemmatizer)...")
        nltk.download('omw-1.4', download_dir=nltk_data_path, quiet=True)

        st.success("NLTK data downloaded and configured successfully!")
        return True # Indicate success
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}. "
                 "Please check internet connection or permissions in the Streamlit Cloud logs.")
        return False # Indicate failure

# Call the NLTK data download function at the start of your app
# Stop the app gracefully if NLTK data download fails
if not download_nltk_data():
    st.stop() 

# Set Streamlit page configuration
st.set_page_config(page_title="News Section Explorer", layout="wide")

# --- Data Loading Function (Cached) ---
# Use st.cache_data to cache the loaded and processed DataFrame.
# This prevents re-running the entire data loading and clustering process
# every time a widget is interacted with, speeding up the app.
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
        return pd.DataFrame(), [] # Return empty DataFrame and list to prevent further errors

    st.info(f"Loading data from {csv_path} and performing clustering. This might take a moment...")
    df, section_names = cluster_articles(csv_path) # cluster_articles reads news.csv and processes it
    st.success("Data loaded and clustered successfully!")
    return df, section_names

# --- Main Streamlit App Layout and Logic ---
def main_app():
    st.title("News Section Clustering and Keyword Explorer")

    # Load data (this function is cached)
    df, section_names = load_data()

    if df.empty:
        st.warning("No data to display. Please check the 'news.csv' file.")
        return

    # Sidebar for filtering/selection
    st.sidebar.header("Filter and Explore")

    # Display basic info
    st.sidebar.write(f"Total articles loaded: {len(df)}")
    st.sidebar.write(f"Number of sections: {len(section_names)}")

    # --- Section Filtering ---
    selected_section = st.sidebar.selectbox(
        "Select a News Section:",
        ["All Sections"] + sorted(section_names.tolist()) # Ensure section_names is a list for sorted()
    )

    filtered_df = df
    if selected_section != "All Sections":
        filtered_df = df[df['section'] == selected_section]
        st.subheader(f"Articles in '{selected_section}' Section ({len(filtered_df)} articles)")
    else:
        st.subheader(f"All News Articles ({len(filtered_df)} articles)")

    # Display filtered articles
    if not filtered_df.empty:
        st.dataframe(filtered_df[['title', 'text', 'section', 'cluster_name']].head(10)) # Display first 10 articles
        if st.checkbox("Show all filtered articles"):
            st.dataframe(filtered_df[['title', 'text', 'section', 'cluster_name']])
    else:
        st.info("No articles found for the selected filter.")


    # --- Display Top Keywords per Section (if clustering results are available) ---
    st.subheader("Top Keywords for Each News Section")

    # Pass the full DataFrame for keyword extraction to ensure all sections are covered
    # (assuming cluster_articles populates 'cluster_name' in the full df)
    if 'processed_text' in df.columns and 'cluster_name' in df.columns:
        top_keywords_dict = get_top_keywords(df, n=10) # n=10 for top 10 keywords

        if top_keywords_dict:
            for section, keywords in top_keywords_dict.items():
                st.write(f"**Section: {section}**")
                # Format keywords for better display
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
