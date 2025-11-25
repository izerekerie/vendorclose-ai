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

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="VendorClose AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
    st.title("🍎 VendorClose AI")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📸 Quick Scan", "📦 Batch Processing", "📊 Dashboard", "🔄 Retraining", "📤 Upload Data"]
    )
    
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


# Main content
st.markdown('<h1 class="main-header">🍎 VendorClose AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Smart End-of-Day Fruit Scanner</p>', unsafe_allow_html=True)

# Quick Scan Page
if page == "📸 Quick Scan":
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
                            
                            # Class and confidence
                            class_name = result['class'].upper()
                            confidence = result['confidence']
                            
                            # Color coding
                            if result['class'] == 'fresh':
                                color = "🟢"
                                st.success(f"{color} **Quality:** {class_name}")
                            elif result['class'] == 'medium':
                                color = "🟡"
                                st.warning(f"{color} **Quality:** {class_name}")
                            else:
                                color = "🔴"
                                st.error(f"{color} **Quality:** {class_name}")
                            
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Action recommendation
                            st.info(f"**Action:** {result['action']}")
                            
                            # Probabilities
                            st.subheader("Class Probabilities")
                            prob_df = pd.DataFrame(
                                list(result['probabilities'].items()),
                                columns=['Class', 'Probability']
                            )
                            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                            st.dataframe(prob_df, use_container_width=True)
                            
                            # Probability bar chart
                            fig = px.bar(
                                prob_df,
                                x='Class',
                                y=prob_df['Probability'].str.rstrip('%').astype('float') / 100,
                                labels={'y': 'Probability'},
                                color='Class',
                                color_discrete_map={
                                    'fresh': 'green',
                                    'medium': 'orange',
                                    'rotten': 'red'
                                }
                            )
                            fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
                        st.info("Make sure the API server is running on port 8000")


# Batch Processing Page
elif page == "📦 Batch Processing":
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
                                results.append({
                                    'Image': pred.get('filename', 'Unknown'),
                                    'Quality': pred['class'].upper(),
                                    'Confidence': f"{pred['confidence']:.2%}",
                                    'Action': pred['action']
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
                            
                            # Group by action
                            sell_now = df[df['Action'].str.contains('Sell now')]
                            keep = df[df['Action'].str.contains('Keep overnight')]
                            remove = df[df['Action'].str.contains('Remove/discard')]
                            
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
elif page == "📊 Dashboard":
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
elif page == "🔄 Retraining":
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
elif page == "📤 Upload Data":
    st.header("📤 Upload Training Data")
    
    st.write("Upload new fruit images to improve the model through retraining.")
    
    # Class selection
    class_label = st.selectbox(
        "Select Fruit Quality Class",
        ["fresh", "medium", "rotten"],
        help="Select the quality class for the images you're uploading"
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

