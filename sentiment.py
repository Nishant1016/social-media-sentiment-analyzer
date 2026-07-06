import os
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Initialize BERTweet-based sentiment model from finiteautomata
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    return tokenizer, model

# Modified sentiment analysis function for the BERTweet-based model
def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Model classes: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        negative, neutral, positive = scores[0].tolist()
        
        # Convert to compound score (-1 to 1 scale)
        compound = (positive - negative)
        return compound

# Set page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bootstrap Integration
def set_bootstrap():
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        """,
        unsafe_allow_html=True,
    )

# Fetch tweets function with fixed infinite loop
def fetch_tweets(word_query, number_of_tweets=10):
    if not BEARER_TOKEN:
        st.error("Bearer token not found. Please set it in your .env file.")
        return pd.DataFrame()

    tokenizer, model = load_sentiment_model()
    
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": word_query,
        "max_results": max(min(number_of_tweets, 100), 10),
        "tweet.fields": "created_at,text",
    }

    max_retries = 3  # Fix: no longer an infinite loop
    attempt = 0

    while attempt < max_retries:
        time.sleep(3)  # Prevent rate limiting
        
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("data", [])

            tweets_df = pd.DataFrame(tweets)
            if not tweets_df.empty:
                tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"])
                tweets_df["sentiment_score"] = tweets_df["text"].apply(
                    lambda x: analyze_sentiment(x, tokenizer, model)
                )

            return tweets_df

        elif response.status_code == 429:
            reset_time = int(response.headers.get("x-rate-limit-reset", time.time() + 60))
            wait_time = reset_time - int(time.time())
            st.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
            time.sleep(wait_time + 1)
            attempt += 1

        else:
            st.error(f"Error fetching tweets: {response.status_code}, {response.json()}")
            return pd.DataFrame()

    st.error("Max retries reached. Please try again later.")
    return pd.DataFrame()

# App function
def app():
    set_bootstrap()
    st.sidebar.header("📊 Analytics Dashboard")
    
    menu_options = ["Sentiment Analysis", "Overview", "Statistics"]
    choice = st.sidebar.radio("Select a section:", menu_options)

    if choice == "Overview":
        st.markdown("<h2 class='text-center'>📈 Social Media Analytics Overview</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        if "tweets_data" in st.session_state:
            _data = st.session_state["tweets_data"]
            _total = len(_data)
            _positive = len(_data[_data["sentiment_score"] > 0.3])
            _negative = len(_data[_data["sentiment_score"] < -0.3])
            _neutral = len(_data[(_data["sentiment_score"] >= -0.3) & (_data["sentiment_score"] <= 0.3)])
            col1.metric("Total Tweets Analyzed", _total)
            col2.metric("Positive Sentiments", _positive)
            col3.metric("Negative Sentiments", _negative)
            col4.metric("Neutral Sentiments", _neutral)
        else:
            col1.metric("Total Tweets Analyzed", "2,450", "+8.5%")
            col2.metric("Positive Sentiments", "1,220", "+12.3%")
            col3.metric("Negative Sentiments", "650", "-5.2%")
            col4.metric("Neutral Sentiments", "580", "+3.1%")
        
        st.markdown("---")

        data = pd.DataFrame({
            "Days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Positive": [500, 600, 700, 550, 750, 620, 500],
            "Negative": [200, 150, 180, 170, 140, 160, 190],
            "Neutral": [300, 320, 310, 290, 305, 310, 315],
        })

        fig = px.line(data, x="Days", y=["Positive", "Negative", "Neutral"], 
                      title="Sentiment Trends Over the Week",
                      labels={"value": "Tweet Count", "Days": "Day of the Week"},
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Statistics":
        st.markdown("<h2 class='text-center'>📊 Sentiment Statistics</h2>", unsafe_allow_html=True)

        if "tweets_data" in st.session_state:
            _data = st.session_state["tweets_data"]
            _positive = len(_data[_data["sentiment_score"] > 0.3])
            _negative = len(_data[_data["sentiment_score"] < -0.3])
            _neutral = len(_data[(_data["sentiment_score"] >= -0.3) & (_data["sentiment_score"] <= 0.3)])
        else:
            _positive, _negative, _neutral = 120, 45, 60

        stats_data = pd.DataFrame({
            "Category": ["Positive Tweets", "Negative Tweets", "Neutral Tweets"],
            "Count": [_positive, _negative, _neutral],
        })

        st.write("### Tweet Sentiment Breakdown")
        st.table(stats_data)

        fig_stats = px.bar(stats_data, x="Category", y="Count", color="Category", title="Sentiment Analysis Statistics")
        st.plotly_chart(fig_stats, use_container_width=True)

    elif choice == "Sentiment Analysis":
        word_query = st.text_input("🔍 Enter a hashtag or keyword:", placeholder="#example")
        number_of_tweets = st.slider("Number of Tweets to Analyze:", min_value=10, max_value=100, step=10)

        if st.button("Analyze Sentiment", key="analyze"):
            if word_query:
                with st.spinner("Fetching tweets and analyzing sentiment..."):
                    try:
                        data = fetch_tweets(word_query, number_of_tweets)

                        if not data.empty:
                            st.session_state["tweets_data"] = data
                            st.subheader("📄 Extracted Dataset")
                            st.dataframe(data)

                            positive = len(data[data["sentiment_score"] > 0.3])
                            neutral = len(data[(data["sentiment_score"] >= -0.3) & (data["sentiment_score"] <= 0.3)])
                            negative = len(data[data["sentiment_score"] < -0.3])

                            sentiment_df = pd.DataFrame({
                                "Sentiment": ["Positive", "Neutral", "Negative"],
                                "Count": [positive, neutral, negative]
                            })

                            fig_sentiment = px.bar(
                                sentiment_df,
                                x="Sentiment",
                                y="Count",
                                color="Sentiment",
                                title="Sentiment Summary"
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)

                        else:
                            st.warning("No tweets found for the given query.")

                    except Exception as e:
                        st.code(f"An error occurred: {e}")
            else:
                st.warning("Please enter a query to fetch tweets.")

if __name__ == "__main__":
    app()
