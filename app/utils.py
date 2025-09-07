import pickle
import pandas as pd
import os
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
from PIL import Image
from io import BytesIO
import base64

def crop_to_aspect(img, aspect_ratio=16/9, position="center"):
    """Crop image ke rasio tertentu. 
    Position: top, center, bottom (default center).
    Untuk gambar portrait (9:16), otomatis ambil bagian tengah-bawah agar POI tidak kepotong.
    """
    w, h = img.size
    target_h = int(w / aspect_ratio)

    if target_h > h:  # Kalau target tinggi lebih besar → crop pakai lebar
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = left + new_w
        top, bottom = 0, h
    else:
        if position == "top":
            left, right = 0, w
            top, bottom = 0, target_h
        elif position == "bottom":
            left, right = 0, w
            # kalau portrait → geser crop lebih ke bawah biar POI (bawah) tetap terlihat
            if h > w:  
                top = int(h * 0.57)   # mulai crop dari 30% tinggi
                bottom = top + target_h
            else:
                top, bottom = h - target_h, h
        else:  # center
            left, right = 0, w
            top = (h - target_h) // 2
            bottom = top + target_h

    return img.crop((left, top, right, bottom))

def add_bg_from_img(img):
    """Convert PIL image ke base64 dan set sebagai background"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Load model dengan path yang benar
def load_model(model_type):
    model_files = {
        "sleep": "xgb_best_model_sleep.pkl",
        "stress": "xgb_best_model_stress.pkl",
        "health": "xgb_best_model_health.pkl"
    }
    if model_type not in model_files:
        raise ValueError(f"Model '{model_type}' tidak ditemukan. Pilih dari: {list(model_files.keys())}")
    model_path = os.path.join("model", model_files[model_type])
    return joblib.load(model_path)

# load scaler
def load_scaler():
    scaler = joblib.load("model/scaler.pkl")
    return scaler

# load encoder
def load_encoders():
    encoder = joblib.load("model/encoders.pkl")
    return encoder

# load encoder
def load_feature_cols():
    encoder = joblib.load("model/feature_cols.pkl")
    return encoder

# Load data sekali
@st.cache_data(show_spinner=True)
def load_data():
    data_path = os.path.join("data", "coffee_health.csv")
    return pd.read_csv(data_path)

coffee_health = load_data()

# Function for feature engineering
def engineer_features(df):
    """
    Performing feature engineering on the training and test sets
    """
    
    if "Health_Issues" in df.columns:
        df["Health_Issues"] = df["Health_Issues"].fillna("None")
        
    age_bins = [0, 30, 40, 50, 60, 100]
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # BMI Categorization
    bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)
    
    # Sleep Hours Categorization
    sleep_bins = [0, 6, 7, 9, 24]
    sleep_labels = ['Deficit', 'Low', 'Recommended', 'Long']
    df['Sleep_Category'] = pd.cut(df['Sleep_Hours'], bins=sleep_bins, labels=sleep_labels)
    
    # 3. Creating New Features
    # Caffeine per Cup
    df['Caffeine_Per_Cup'] = df['Caffeine_mg'] / df['Coffee_Intake'].replace(0, 1)
    
    # Interaction Features
    df['BMI_Activity_Interaction'] = df['BMI'] * df['Physical_Activity_Hours']
    
    df['Caffeine_Sleep_Interaction'] = df['Caffeine_mg'] * df['Sleep_Hours']
    
    return df

def get_dynamic_palette(n):
            """Generate palette: if n>3 use darkening gradient, else fixed Set2."""
            if n <= 3:
                return sns.color_palette("Set2", n)
            else:
                # gradasi dari terang ke gelap (Blues)
                return sns.color_palette("Blues", n)


# ==== helpers (taruh di atas file, sekali saja) ====
def safe_transform(le, s):
    """Transform dengan fallback jika ada label baru yang tidak dikenal encoder."""
    s = s.astype(str)
    known = set(le.classes_)
    if not s.isin(known).all():
        # fallback pakai kelas paling sering (idx 0 di LabelEncoder biasanya alfabetis;
        # kalau mau yang paling sering beneran, simpan saat training)
        fallback = le.classes_[0]
        s = s.where(s.isin(known), fallback)
    return le.transform(s)

def apply_feature_encoders(df, encoders_features, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            le = encoders_features.get(col)
            if le is None:
                # kalau kamu masih punya model lama (flat dict), fallback:
                le = encoders_features.get("features", {}).get(col)
            if le is None:
                # kalau tetap None, skip
                # bisa juga st.warning, tapi jangan ngehentikan eksekusi
                continue
            df[col] = safe_transform(le, df[col])
    return df


