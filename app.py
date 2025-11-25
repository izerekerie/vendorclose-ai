"""
Streamlit UI for VendorClose AI
Web interface for predictions, visualizations, and model management
"""

import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Add src to path
sys.path.append(str(Path(__file__).parent))

# API configuration - Use environment variable for production
# Set in Streamlit Cloud secrets or Render environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="VendorClose AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode with High Contrast
st.markdown("""
    <style>
    /* Dark Mode Base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f7812a;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #2d3142;
    }
    
    /* Sidebar - Dark Mode */
    section[data-testid="stSidebar"] {
        background-color: #1e2130 !important;
        border-right: 1px solid #2d3142;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #1e2130 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p {
        color: #fafafa !important;
    }
    
    /* Navigation - Dark Mode with High Contrast */
    div[data-testid="stRadio"] > div {
        background-color: #1e2130 !important;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 2px solid #2d3142 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stRadio"] > div:hover {
        background-color: #2d3142 !important;
        border-color: #f7812a !important;
        transform: translateX(5px);
    }
    
    /* Selected radio button - Orange/High Contrast */
    div[data-testid="stRadio"] > div[aria-checked="true"] {
        background-color: #f7812a !important;
        border-color: #f7812a !important;
        color: #0e1117 !important;
        font-weight: bold;
    }
    
    /* Radio button labels - Always visible in dark mode */
    div[data-testid="stRadio"] > div > label {
        color: #fafafa !important;
        font-size: 1rem;
        font-weight: 500;
        padding: 0.5rem;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"] > div[aria-checked="true"] > label {
        color: #0e1117 !important;
        font-weight: bold;
    }
    
    /* Radio button circle - Visible */
    div[data-testid="stRadio"] div[data-baseweb="radio"] {
        border: 2px solid #f7812a;
        background-color: #1e2130;
    }
    
    div[data-testid="stRadio"] div[aria-checked="true"] div[data-baseweb="radio"] {
        background-color: #0e1117 !important;
        border-color: #0e1117 !important;
    }
    
    /* All text elements - Dark mode */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Input fields - Dark mode */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #1e2130 !important;
        color: #fafafa !important;
        border-color: #2d3142 !important;
    }
    
    /* Buttons - Dark mode */
    .stButton > button {
        background-color: #f7812a !important;
        color: #0e1117 !important;
        border: none;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #ff9a4d !important;
    }
    
    /* File uploader - Dark mode */
    .uploadedFile {
        background-color: #1e2130 !important;
        border-color: #2d3142 !important;
    }
    
    /* Dataframes - Dark mode */
    .dataframe {
        background-color: #1e2130 !important;
        color: #fafafa !important;
    }
    
    /* Hide app icon if too prominent */
    .stApp > header {
        visibility: hidden;
        height: 0;
    }
    
    /* Status indicators */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #1e2130 !important;
        border-left: 4px solid;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_model_uptime():
    """Calculate model uptime (simplified)"""
    # In production, this would track actual uptime
    if check_api_health():
        return "🟢 Online"
    return "🔴 Offline"


# Sidebar
with st.sidebar:
    # Header with basket icon instead of apple
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem; margin: 0;">🛒 VendorClose AI</h1>
        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0;">Smart Fruit Scanner</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation - Standard web app style with visible links
    st.markdown("### 🧭 Navigation")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Quick Scan"
    
    # Navigation options - clean text without emojis for better visibility
    nav_options = {
        "Quick Scan": "📸",
        "Batch Processing": "📦",
        "Dashboard": "📊",
        "Retraining": "🔄",
        "Upload Data": "📤"
    }
    
    # Create navigation with visible buttons
    nav_items = list(nav_options.keys())
    current_index = nav_items.index(st.session_state.current_page) if st.session_state.current_page in nav_items else 0
    
    selected = st.radio(
        "Choose a page",
        nav_items,
        index=current_index,
        label_visibility="visible"
    )
    
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()
    
    page = f"{nav_options[selected]} {selected}" if selected in nav_options else selected
    
    st.markdown("---")
    
    # Model Status
    st.subheader("Model Status")
    uptime = get_model_uptime()
    st.markdown(f"**Status:** {uptime}")
    
    if check_api_health():
        try:
            stats = requests.get(f"{API_BASE_URL}/stats", timeout=2).json()
            st.markdown(f"**Model Loaded:** {'✅' if stats.get('model_loaded') else '❌'}")
            st.markdown(f"**Total Images:** {stats.get('total_images', 0)}")
        except:
            pass


# Main content - Dark mode styling
st.markdown('<h1 class="main-header">🛒 VendorClose AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #b0b0b0;">Smart End-of-Day Fruit Scanner</p>', unsafe_allow_html=True)

# Quick Scan Page
if "Quick Scan" in page or page == "📸 Quick Scan":
    st.header("📸 Quick Scan - Single Fruit Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a fruit image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a single fruit image to get instant quality assessment"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            if st.button("🔍 Analyze Fruit", type="primary"):
                with st.spinner("Analyzing fruit quality..."):
                    try:
                        # Prepare file for API
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=10)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.subheader("Analysis Results")
                            
                            # Class and confidence - handle both 'class' and 'class_name'
                            class_name = result.get('class_name', result.get('class', 'unknown'))
                            if isinstance(class_name, str):
                                class_name = class_name.upper()
                            confidence = result.get('confidence', 0.0)
                            
                            # Color coding - supports all 19 classes
                            class_lower = str(class_name).lower()
                            if 'fresh' in class_lower:
                                color = "🟢"
                                st.success(f"{color} **Quality:** {class_name}")
                            elif 'medium' in class_lower:
                                color = "🟡"
                                st.warning(f"{color} **Quality:** {class_name}")
                            elif 'rotten' in class_lower:
                                color = "🔴"
                                st.error(f"{color} **Quality:** {class_name}")
                            else:
                                st.info(f"**Quality:** {class_name}")
                            
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Action recommendation
                            action = result.get('action', 'No recommendation available')
                            st.info(f"**Action:** {action}")
                            
                            # Probabilities - handle 19 classes
                            probabilities = result.get('probabilities', {})
                            if probabilities:
                                st.subheader("Class Probabilities")
                                prob_df = pd.DataFrame(
                                    list(probabilities.items()),
                                    columns=['Class', 'Probability']
                                )
                                # Convert probability values
                                prob_df['Probability'] = prob_df['Probability'].apply(
                                    lambda x: float(x) if isinstance(x, (int, float)) else 0.0
                                )
                                prob_df['Probability_Display'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                                
                                # Sort by probability
                                prob_df = prob_df.sort_values('Probability', ascending=False)
                                st.dataframe(prob_df[['Class', 'Probability_Display']].rename(columns={'Probability_Display': 'Probability'}), use_container_width=True)
                                
                                # Probability bar chart - color by quality
                                def get_color(class_name):
                                    class_lower = str(class_name).lower()
                                    if 'fresh' in class_lower:
                                        return 'green'
                                    elif 'medium' in class_lower:
                                        return 'orange'
                                    elif 'rotten' in class_lower:
                                        return 'red'
                                    return 'gray'
                                
                                prob_df['Color'] = prob_df['Class'].apply(get_color)
                                
                                fig = px.bar(
                                    prob_df.head(10),  # Show top 10
                                    x='Class',
                                    y='Probability',
                                    labels={'Probability': 'Probability'},
                                    color='Color',
                                    color_discrete_map='identity'
                                )
                                fig.update_layout(showlegend=False, yaxis_tickformat='.0%', height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
                        st.info("Make sure the API server is running on port 8000")


# Batch Processing Page
elif "Batch Processing" in page or page == "📦 Batch Processing":
    st.header("📦 Batch Processing - Multiple Fruits")
    
    uploaded_files = st.file_uploader(
        "Upload multiple fruit images (up to 50)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images to analyze at once"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 50:
            st.warning("⚠️ Maximum 50 images allowed. Processing first 50.")
            uploaded_files = uploaded_files[:50]
        
        if st.button("🔍 Analyze All Fruits", type="primary"):
            with st.spinner(f"Analyzing {len(uploaded_files)} fruits..."):
                results = []
                files_data = []
                
                for file in uploaded_files:
                    files_data.append(('files', (file.name, file.getvalue(), file.type)))
                
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict/batch",
                        files=files_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        predictions = response.json()['predictions']
                        
                        # Process results
                        for pred in predictions:
                            if 'error' not in pred:
                                class_name = pred.get('class_name', pred.get('class', 'Unknown'))
                                results.append({
                                    'Image': pred.get('filename', 'Unknown'),
                                    'Quality': str(class_name).upper(),
                                    'Confidence': f"{pred.get('confidence', 0.0):.2%}",
                                    'Action': pred.get('action', 'No recommendation')
                                })
                        
                        # Display results table
                        if results:
                            df = pd.DataFrame(results)
                            st.dataframe(df, use_container_width=True)
                            
                            # Statistics
                            st.subheader("📊 Batch Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            quality_counts = df['Quality'].value_counts()
                            
                            with col1:
                                st.metric("Total Analyzed", len(results))
                            with col2:
                                fresh_count = quality_counts.get('FRESH', 0)
                                st.metric("Fresh", fresh_count, f"{fresh_count/len(results):.1%}")
                            with col3:
                                medium_count = quality_counts.get('MEDIUM', 0)
                                st.metric("Medium", medium_count, f"{medium_count/len(results):.1%}")
                            with col4:
                                rotten_count = quality_counts.get('ROTTEN', 0)
                                st.metric("Rotten", rotten_count, f"{rotten_count/len(results):.1%}")
                            
                            # Priority action list
                            st.subheader("🎯 Priority Action List")
                            
                            # Group by action - handle 19 classes
                            sell_now = df[df['Action'].str.contains('Sell now', case=False, na=False)]
                            keep = df[df['Action'].str.contains('Keep overnight', case=False, na=False)]
                            remove = df[df['Action'].str.contains('Remove|discard', case=False, na=False)]
                            
                            if len(remove) > 0:
                                st.error(f"❌ **Remove/Discard ({len(remove)}):** Priority - Remove these immediately")
                                st.dataframe(remove[['Image', 'Quality', 'Confidence']], use_container_width=True)
                            
                            if len(sell_now) > 0:
                                st.warning(f"⚠️ **Sell Now with Discount ({len(sell_now)}):** Sell these today")
                                st.dataframe(sell_now[['Image', 'Quality', 'Confidence']], use_container_width=True)
                            
                            if len(keep) > 0:
                                st.success(f"✅ **Keep Overnight ({len(keep)}):** Still fresh for tomorrow")
                                st.dataframe(keep[['Image', 'Quality', 'Confidence']], use_container_width=True)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# Dashboard Page
elif "Dashboard" in page or page == "📊 Dashboard":
    st.header("📊 Business Dashboard")
    
    if not check_api_health():
        st.error("⚠️ API is not available. Please start the API server.")
    else:
        try:
            # Get statistics
            stats = requests.get(f"{API_BASE_URL}/stats", timeout=2).json()
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", stats.get('total_images', 0))
            with col2:
                st.metric("Used for Training", stats.get('used_images', 0))
            with col3:
                st.metric("Available for Training", stats.get('unused_images', 0))
            with col4:
                st.metric("Training Sessions", stats.get('total_sessions', 0))
            
            st.markdown("---")
            
            # Class distribution
            st.subheader("📈 Data Distribution by Class")
            images_by_class = stats.get('images_by_class', {})
            
            if images_by_class:
                class_df = pd.DataFrame(
                    list(images_by_class.items()),
                    columns=['Class', 'Count']
                )
                
                fig = px.pie(
                    class_df,
                    values='Count',
                    names='Class',
                    color='Class',
                    color_discrete_map={
                        'fresh': 'green',
                        'medium': 'orange',
                        'rotten': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart
                fig2 = px.bar(
                    class_df,
                    x='Class',
                    y='Count',
                    color='Class',
                    color_discrete_map={
                        'fresh': 'green',
                        'medium': 'orange',
                        'rotten': 'red'
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Training sessions
            st.subheader("🔄 Recent Training Sessions")
            sessions = requests.get(f"{API_BASE_URL}/sessions", timeout=2).json().get('sessions', [])
            
            if sessions:
                sessions_df = pd.DataFrame(sessions)
                st.dataframe(sessions_df, use_container_width=True)
            else:
                st.info("No training sessions yet")
            
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")


# Retraining Page
elif "Retraining" in page or page == "🔄 Retraining":
    st.header("🔄 Model Retraining")
    
    if not check_api_health():
        st.error("⚠️ API is not available. Please start the API server.")
    else:
        # Get retraining status
        try:
            status = requests.get(f"{API_BASE_URL}/retrain/status", timeout=2).json()
            
            st.subheader("Current Status")
            status_col = status['status']
            
            if status_col == "idle":
                st.info("✅ Ready for retraining")
            elif status_col == "in_progress":
                st.warning("⏳ Retraining in progress...")
            elif status_col == "completed":
                st.success("✅ Retraining completed!")
            elif status_col == "failed":
                st.error("❌ Retraining failed")
            
            # Progress bar
            if status_col == "in_progress":
                st.progress(status['progress'] / 100)
                st.write(f"Progress: {status['progress']}%")
            
            st.write(f"**Message:** {status['message']}")
            
            if status['session_id']:
                st.write(f"**Session ID:** {status['session_id']}")
            
            st.markdown("---")
            
            # Trigger retraining
            st.subheader("Trigger Retraining")
            st.write("Click the button below to start retraining with newly uploaded data.")
            
            if st.button("🚀 Start Retraining", type="primary"):
                with st.spinner("Starting retraining..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/retrain", timeout=5)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.info(f"Session ID: {result['session_id']}")
                            st.rerun()
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Auto-refresh if training
            if status_col == "in_progress":
                time.sleep(2)
                st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Upload Data Page
elif "Upload Data" in page or page == "📤 Upload Data":
    st.header("📤 Upload Training Data")
    
    st.write("Upload new fruit images to improve the model through retraining.")
    
    # Class selection - All 19 classes supported
    all_classes = [
        'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum',
        'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
        'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum',
        'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato',
        'medium'
    ]
    
    def format_class_name(class_name):
        """Format class name for display"""
        if class_name == 'medium':
            return 'Medium Quality'
        name = class_name.replace('fresh', 'Fresh ').replace('rotten', 'Rotten ')
        # Capitalize first letter of fruit name
        parts = name.split()
        if len(parts) > 1:
            parts[1] = parts[1].capitalize()
        return ' '.join(parts)
    
    class_label = st.selectbox(
        "Select Fruit Quality Class",
        all_classes,
        help="Select the specific fruit type and quality for the images you're uploading",
        format_func=format_class_name
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload training images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images for this quality class"
    )
    
    if uploaded_files:
        st.write(f"**Selected class:** {class_label}")
        st.write(f"**Number of files:** {len(uploaded_files)}")
        
        # Preview images
        if st.checkbox("Preview images"):
            cols = st.columns(min(4, len(uploaded_files)))
            for idx, file in enumerate(uploaded_files[:8]):  # Show first 8
                with cols[idx % 4]:
                    image = Image.open(file)
                    st.image(image, use_container_width=True)
                    st.caption(file.name)
        
        # Upload button
        if st.button("📤 Upload Images", type="primary"):
            with st.spinner(f"Uploading {len(uploaded_files)} images..."):
                files_data = []
                for file in uploaded_files:
                    files_data.append(('files', (file.name, file.getvalue(), file.type)))
                
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/upload?class_label={class_label}",
                        files=files_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"Uploaded {result['files_uploaded']} images for class '{result['class_label']}'")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

