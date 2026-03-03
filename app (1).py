import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai
from google.genai import types
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# ----------------------------------
# 🔧 Configuration & Page Setup
# ----------------------------------
st.set_page_config(page_title="RES-Q AI Disaster Response", page_icon="🚨", layout="wide")

# ----------------------------------
# 🗺️ Location Finder Function
# ----------------------------------
def find_coordinates(location_name):
    """
    Converts location name to latitude/longitude using OpenStreetMap (Free)
    """
    geolocator = Nominatim(user_agent="resq_ai_app")
    
    # Common location mappings for better accuracy
    location_mappings = {
        "usa": "United States", "us": "United States",
        "uk": "United Kingdom", "uae": "United Arab Emirates",
        "india": "India", "chennai": "Chennai, Tamil Nadu, India",
        "mumbai": "Mumbai, Maharashtra, India",
        "delhi": "New Delhi, India",
        "california": "California, USA",
        "new york": "New York City, USA",
        "texas": "Texas, USA",
        "florida": "Florida, USA",
        "japan": "Japan",
        "indonesia": "Indonesia",
        "turkey": "Turkey",
        "syria": "Syria",
        "brazil": "Brazil",
        "australia": "Australia",
        "ukraine": "Ukraine",
        "russia": "Russia",
        "china": "China",
    }
    
    search_name = location_mappings.get(location_name.lower(), location_name)
    
    try:
        location = geolocator.geocode(search_name)
        if location:
            return location.latitude, location.longitude, location.address
    except GeocoderTimedOut:
        return None, None, None
    except Exception:
        return None, None, None
    
    return None, None, None

def extract_location_from_text(text):
    """
    Simple keyword-based location extraction from text
    """
    text = text.lower()
    
    # Common disaster-prone locations keywords
    locations = [
        "chennai", "mumbai", "delhi", "kolkata", "bangalore",
        "california", "texas", "florida", "new york", "los angeles",
        "japan", "turkey", "syria", "indonesia", "nepal",
        "pakistan", "brazil", "australia", "ukraine", "russia",
        "china", "india", "usa", "uk", "uae", "saudi arabia",
        "haiti", "mexico", "peru", "chile", "italy", "greece"
    ]
    
    found_locations = []
    for loc in locations:
        if loc in text:
            found_locations.append(loc)
    
    return found_locations[0] if found_locations else None

# ----------------------------------
# 🧠 ML Model Loader (Cached)
# ----------------------------------
@st.cache_resource
def load_ml_model():
    try:
        data = pd.read_csv(r"D:\RES-Q-AI\Dataset\train.csv")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        data['text'] = data['text'].apply(clean_text)

        X = data['text']
        y = data['target']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.27, random_state=40)

        vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.8
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf   = vectorizer.transform(X_val)

        clf = LogisticRegression(max_iter=2500, solver='liblinear', class_weight='balanced')
        clf.fit(X_train_tfidf, y_train)
        
        return vectorizer, clf, clean_text
    except FileNotFoundError:
        st.error("Error: train.csv not found.")
        return None, None, None

# ----------------------------------
# 🌐 Gemini Integration
# ----------------------------------
DEMO_RESPONSES = {
    "current": """
## 🆘 Live Emergency Analysis

### 🔍 Disaster Type: Flood
### 🚨 Severity: HIGH

### 📰 Latest News:
Heavy rainfall has been reported in the mentioned area. Local authorities have issued a flood warning.

### ⚠️ Safety Precautions:
- Move to higher ground immediately
- Avoid walking in standing water
- Keep emergency supplies ready

### 🆘 Emergency Action Plan:
1. Evacuate if instructed
2. Call emergency services if trapped
3. Do not return until officials give all-clear
""",
    "predict": """
## 🌍 Risk Assessment

### 🔮 Prediction: High Likelihood of Occurrence

### 🚦 Risk Level: HIGH

### 📰 Current Conditions:
Weather alerts indicate favorable conditions for disaster occurrence.

### ⚠️ Safety Cautions:
- Prepare emergency kit
- Stay updated with local news
- Have evacuation plan ready
"""
}

def get_gemini_response(user_input, mode, api_key, use_demo=False):
    if use_demo:
        return DEMO_RESPONSES.get(mode, "Demo response not available.")

    if not api_key:
        st.warning("Please enter your Google AI API Key.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return None

    prompt = ""
    if mode == "current":
        prompt = f"""A tweet has been detected as a REAL disaster: "{user_input}". Classify disaster type, severity, search latest news, provide safety precautions, and identify the location."""
    else:
        prompt = f"""Predict if a disaster is likely for: "{user_input}". Provide risk level and safety precautions."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini: {e}")
        return None

# ----------------------------------
# 🎨 UI Layout
# ----------------------------------
st.sidebar.title("🔑 Configuration")

use_demo = st.sidebar.checkbox("🧪 Use Demo Mode (No API Key)", value=False)
api_key = None

if not use_demo:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    st.sidebar.markdown("*Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)*")
else:
    st.sidebar.info("Running in Demo Mode")

# Main App
st.title("🚨 RES-Q AI - Disaster Detection System")
st.markdown("### Real-time Tweet Analysis & Disaster Prediction with Live Map")

vectorizer, clf, clean_text_func = load_ml_model()

if vectorizer and clf:
    mode = st.radio("Choose Mode:", 
                    ["1. Analyze Tweet (ML + Live Data)", "2. Disaster Prediction (Live Data Only)"], 
                    horizontal=True)

    st.divider()

    if "1." in mode:
        st.subheader("📝 Tweet Analysis")
        tweet_input = st.text_area("Enter tweet to analyze:", height=100, 
                                   placeholder="e.g., 'Severe flooding in Chennai, many houses destroyed...'")
        
        if st.button("Analyze Tweet", type="primary"):
            if tweet_input:
                with st.spinner("Analyzing with ML Model..."):
                    cleaned = clean_text_func(tweet_input)
                    tweet_vec = vectorizer.transform([cleaned])
                    prob = clf.predict_proba(tweet_vec)[0][1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Disaster Probability", f"{round(prob * 100, 2)}%")
                    with col2:
                        if prob > 0.65:
                            st.success("🚨 REAL DISASTER DETECTED")
                        else:
                            st.info("✅ NOT A DISASTER")

                    # --- 🗺️ MAP SECTION ---
                    st.subheader("🗺️ Location Map")
                    
                    # Extract location from tweet
                    extracted_loc = extract_location_from_text(tweet_input)
                    
                    if extracted_loc:
                        with st.spinner(f"Finding location: {extracted_loc}..."):
                            lat, lon, address = find_coordinates(extracted_loc)
                            
                            if lat and lon:
                                st.success(f"📍 Location Found: {address}")
                                
                                # Create DataFrame for map
                                map_data = pd.DataFrame({
                                    'lat': [lat],
                                    'lon': [lon]
                                })
                                
                                # Display map
                                st.map(map_data, zoom=5)
                                
                                # Show coordinates
                                st.code(f"Latitude: {lat}, Longitude: {lon}")
                            else:
                                st.warning("Location found but couldn't get coordinates.")
                    else:
                        st.info("ℹ️ No specific location detected in the tweet. You can enter location manually below.")
                        
                        # Manual location input
                        manual_loc = st.text_input("Enter location manually:", placeholder="e.g., California, Japan, Mumbai")
                        if manual_loc:
                            with st.spinner("Finding location..."):
                                lat, lon, address = find_coordinates(manual_loc)
                                if lat and lon:
                                    st.success(f"📍 Location: {address}")
                                    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                                    st.map(map_data, zoom=5)
                                    st.code(f"Latitude: {lat}, Longitude: {lon}")
                    # ----------------------

                    # Get Gemini/Demo response
                    st.divider()
                    result = get_gemini_response(tweet_input, "current", api_key, use_demo)
                    if result:
                        st.markdown("### 🆘 Analysis Result")
                        st.markdown(result)
            else:
                st.warning("Please enter some text.")

    elif "2." in mode:
        st.subheader("🔮 Disaster Prediction")
        pred_input = st.text_input("Location / Situation", placeholder="e.g., California wildfire risk")
        
        if st.button("Predict Risk", type="primary"):
            if pred_input:
                with st.spinner("Analyzing..."):
                    # Try to find location on map
                    extracted_loc = extract_location_from_text(pred_input)
                    
                    if extracted_loc:
                        lat, lon, address = find_coordinates(extracted_loc)
                        if lat and lon:
                            st.success(f"📍 Monitoring: {address}")
                            map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                            st.map(map_data, zoom=4)

                    result = get_gemini_response(pred_input, "predict", api_key, use_demo)
                    if result:
                        st.markdown("### 🌍 Risk Assessment")
                        st.markdown(result)
            else:
                st.warning("Please enter a location.")