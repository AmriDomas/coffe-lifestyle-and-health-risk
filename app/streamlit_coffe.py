# app.py (fixed)
import streamlit as st

# ===== MUST be the first Streamlit command =====
st.set_page_config(page_title="Coffe Life Style and Health Risk", layout="wide")
# =================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import requests
import os
import tempfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import load_model, load_data, engineer_features, get_dynamic_palette, load_encoders, load_scaler, safe_transform, apply_feature_encoders, load_feature_cols, crop_to_aspect, add_bg_from_img

from PIL import Image
from io import BytesIO

# Load data
coffee_health = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Exploratory Data Analysis", "Predictive Modeling"])
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Muh Amri Sidiq")
st.sidebar.markdown("Data Source: [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/coffee-health-and-lifestyle)")

# Home Page
if app_mode == "Home":
    # Load dan crop gambar ke 16:9 bagian bawah
    url = "https://images.unsplash.com/photo-1509042239860-f550ce710b93?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_cropped = crop_to_aspect(img, aspect_ratio=16/9, position="bottom")

    # Set background
    add_bg_from_img(img_cropped)

    # Konten utama
    st.title("☕ Coffee Life Style and Health Risk")
    st.markdown("""
    Welcome to the **Coffee Life Style and Health Risk** application!  

    This app helps you explore how coffee consumption connects with **sleep quality, stress levels, and health risks**.

    ### 🚀 Features:
    - 📊 **Exploratory Data Analysis**: Interactive plots to explore patterns.  
    - 🤖 **Predictive Modeling**: ML models that predict sleep quality, stress level, and health risks from your lifestyle.  

    ### 📝 Instructions:
    - Use the sidebar to navigate across sections.  
    - In *Predictive Modeling*, enter your personal data for predictions.  

    ☕ **Enjoy discovering how coffee affects your lifestyle & health!**
    """)

# Exploratory Data Analysis Page
elif app_mode == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Feature Engineering
    df = engineer_features(coffee_health.copy())

    # Tab setup
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Preview", "📈 Stats", "🔥 Correlation", "🎨 Custom Plot"
    ])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Basic Statistics")
        st.write(df.describe(include='all'), use_container_width=True)


    def plot_categorical(df, col):
        num_cat = df[col].nunique(dropna=False)
        palette = get_dynamic_palette(num_cat)

        fig, ax = plt.subplots(figsize=(6, 3))
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df, order=order, palette=palette, ax=ax)

        # Title & ticks lebih proporsional
        ax.set_title(f"Distribusi {col}", fontsize=8, weight='bold')
        ax.set_xlabel(col, fontsize=6)  # benerin param
        ax.set_ylabel("Count", fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        ax.tick_params(axis='y', labelsize=5)

        # Label bar lebih kecil, posisinya di edge
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        sns.despine()  # buang border luar
        st.pyplot(fig)


    with tab2:

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("📊 Distribution Category")
        # exclude hanya kolom 'Id' dan 'Timestamp' (persis nama itu)
        exclude_cols = ["date"]

        kategori_cols = [c for c in df.select_dtypes(include='object').columns if c not in exclude_cols]

        col_kategori = st.selectbox("Select the category column", kategori_cols)

        col1, col2 = st.columns([1,1])

        with col1:
            st.subheader("📊 Distribusi Kategori (Bar Chart)")
            fig, ax = plt.subplots(figsize=(5, 4))  

            order = df[col_kategori].value_counts().index
            num_cat = len(order)
            palette = get_dynamic_palette(num_cat)

            sns.countplot(
                x=col_kategori, 
                data=df, 
                order=order, 
                palette=palette, 
                width=0.6,  # lebih ramping
                ax=ax
            )

            ax.set_title(f"Distribution {col_kategori}", fontsize=10, weight='bold')
            ax.set_xlabel(col_kategori, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=6, label_type='edge', padding=1)

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Distrbution Category (Pie Chart)")
            val_counts = df[col_kategori].value_counts()

            # ukuran fix biar gak berubah-ubah
            fig, ax = plt.subplots(figsize=(5, 4))  

            wedges, texts, autotexts = ax.pie(
                val_counts,
                labels=None,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("Blues", len(val_counts))
            )

            # kunci aspect ratio -> pie selalu lingkaran sempurna
            ax.axis('equal')

            percentages = val_counts / val_counts.sum() * 100
            ax.legend(
                wedges,
                [f"{cat} ({p:.1f}%)" for cat, p in zip(val_counts.index, percentages)],
                title=col_kategori,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=6,
                title_fontsize=8
            )
            
            plt.tight_layout()
            st.pyplot(fig)


    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.drop(columns=["ID"], errors='ignore').corr(numeric_only=True)
        sns.heatmap(
            corr,
            annot=True,
            cmap="Spectral",
            ax=ax,
            annot_kws={"size": 8}   # angka dalam kotak
        )
        ax.set_title("Correlation Heatmap", fontsize=8)

        # atur ukuran nama kolom & baris
        ax.tick_params(axis='x', labelsize=7, rotation=90)  # label kolom
        ax.tick_params(axis='y', labelsize=7, rotation=0)   # label baris

        # --- Adjust tulisan legend (colorbar) ---
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)   # ukuran angka di legend
        cbar.set_label("Correlation Ratio", fontsize=8)  # label legend + fontsize

        st.pyplot(fig)

    with tab4:
        st.subheader("📊 Custom Plot (Auto Bar / Scatter / Box)")
        all_cols = [c for c in df.columns if c.lower() != 'date']
        col_x = st.selectbox("Select Column X", all_cols, index=0)
        col_y = st.selectbox("Select Column Y", all_cols, index=1)

        fig, ax = plt.subplots(figsize=(8, 3))
        x_is_cat = df[col_x].dtype == 'object'
        y_is_cat = df[col_y].dtype == 'object'

        if x_is_cat and y_is_cat:
            num_cat = df[col_y].nunique(dropna=False)
            palette = get_dynamic_palette(num_cat)
            sns.countplot(x=col_x, hue=col_y, data=df, palette=palette, ax=ax)
            ax.set_title(f"Bar Chart: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel("Count", fontsize=6)
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        elif not x_is_cat and not y_is_cat:
            sns.scatterplot(x=col_x, y=col_y, data=df, color="royalblue", ax=ax)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel(col_y, fontsize=6)
            ax.tick_params(axis='both', labelsize=5)

        else:
            if x_is_cat:
                num_cat = df[col_x].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_x, y=col_y, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_x, fontsize=6)
                ax.set_ylabel(col_y, fontsize=6)
            else:
                num_cat = df[col_y].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_y, y=col_x, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_y, fontsize=6)
                ax.set_ylabel(col_x, fontsize=6)

            ax.set_title(f"Box Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        # Atur legend biar kecil
        leg = ax.get_legend()
        if leg:
            leg.set_title(leg.get_title().get_text(), prop={'size': 5})
            for text in leg.get_texts():
                text.set_fontsize(5)

        sns.despine()
        st.pyplot(fig)

# ====== PREDICTION ======
elif app_mode == "Predictive Modeling":
    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    T1, T2, T3, T4 = st.tabs([
        "Clustering Explorer", "Predictive Model", "Policy Simulation", "Business Insights"
    ])

    df = engineer_features(coffee_health.copy())
    coffe = df.copy()

    with T1:
        st.subheader("Clustering Explorer")
        features = ["Coffee_Intake", "Sleep_Hours", "BMI", "Stress_Level", "Sleep_Quality"]
        coffe_cluster = df[features].copy()

        ordinal_maps = {
            "Stress_Level": {"Low": 0, "Medium": 1, "High": 2},
            "Sleep_Quality": {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
        }
        for col, mapping in ordinal_maps.items():
            coffe_cluster[col] = coffe_cluster[col].map(mapping)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(coffe_cluster)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        coffe["Cluster"] = cluster_labels
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(scaled_data)
        coffe["PCA1"], coffe["PCA2"] = pca_data[:,0], pca_data[:,1]

        fig, ax = plt.subplots(figsize=(4, 2))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=coffe, palette="Set2", ax=ax, s=10)

        # Tambahkan centroid dengan label cluster
        centroids = pca.transform(kmeans.cluster_centers_)
        for i, (x, y) in enumerate(centroids):
            ax.text(x, y, str(i), fontsize=4, weight='bold',
                    color="black", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.3, edgecolor="gray"))
        
        legend = ax.legend(title="Cluster", title_fontsize=4, fontsize=3, loc="best", frameon=True)
        plt.setp(legend.get_title(), weight="bold")

        # Tambahkan judul dan label sumbu
        ax.set_title("PCA Clustering of Coffee & Health Patterns", fontsize=6)
        ax.set_xlabel("Principal Component 1", fontsize=4)
        ax.set_ylabel("Principal Component 2", fontsize=4)

        # ✅ Adjust ukuran angka di sumbu (ticks)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)

        st.pyplot(fig)

        cluster_summary = coffe_cluster.assign(Cluster=cluster_labels).groupby("Cluster").mean()
        st.write("Cluster Summary")
        st.dataframe(cluster_summary)


    with T2:

        # Load models + encoders + scaler
        sleep_model = load_model("sleep")
        stress_model = load_model("stress")
        health_model = load_model("health")
        encoders = load_encoders()
        scaler = load_scaler()
        feature_cols = load_feature_cols()

        # Form Input
        with st.form("prediction_form"):
            age = st.number_input("Age", min_value=10, max_value=90, value=30)
            gender = st.selectbox("Gender", encoders["features"]["Gender"].classes_)
            country = st.selectbox("Country", encoders["features"]["Country"].classes_)
            occupation = st.selectbox("Occupation", encoders["features"]["Occupation"].classes_)
            coffee_intake = st.number_input("Coffee Intake (cups/day)", min_value=0, max_value=15, value=2)
            caffeine_mg = st.number_input("Caffeine (mg/day)", min_value=0, max_value=1000, value=150)
            sleep_hours = st.number_input("Sleep Hours", min_value=3, max_value=12, value=7)
            bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.0)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70)
            physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=40, value=5)
            alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

            # ✅ Tambahkan tombol submit
            submit = st.form_submit_button("Predict")

        if submit:
            # Raw input dataframe
            df_raw = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Country": country,
                "Occupation": occupation,
                "Coffee_Intake": coffee_intake,
                "Caffeine_mg": caffeine_mg,
                "Sleep_Hours": sleep_hours,
                "BMI": bmi,
                "Heart_Rate": heart_rate,
                "Physical_Activity_Hours": physical_activity,
                "Alcohol_Consumption": alcohol,
                "Smoking": smoking
            }])

            # 1. Feature engineering
            df_feat = engineer_features(df_raw)

            # 2. Encode kategorikal
            cat_cols = ["Gender", "Country", "Occupation", 
                        "Age_Group", "BMI_Category", "Sleep_Category"]

            df_feat = apply_feature_encoders(df_feat, encoders["features"], cat_cols)

            df_feat = df_feat.reindex(columns=feature_cols)

            # 5) Scale
            df_scaled = scaler.transform(df_feat)

            # ✅ 4. Prediction
            pred_sleep = sleep_model.predict(df_scaled)[0]
            pred_stress = stress_model.predict(df_scaled)[0]
            pred_health = health_model.predict(df_scaled)[0]

            # ✅ 5. Decode kembali label prediksi
            label_sleep = encoders["targets"]['Sleep_Quality'].inverse_transform([pred_sleep])[0]
            label_stress = encoders["targets"]['Stress_Level'].inverse_transform([pred_stress])[0]
            label_health = encoders["targets"]['Health_Issues'].inverse_transform([pred_health])[0]

            # ✅ 6. Display result
            st.write("### Prediction Results")
            st.write(f"**Sleep Quality:** {label_sleep}")
            st.write(f"**Stress Level:** {label_stress}")
            st.write(f"**Health Issues:** {label_health}")

    with T3:
        st.subheader("Policy Simulation")
        avg_before = coffe["Sleep_Hours"].mean()
        reduction = st.slider("Reduce Coffee Intake (cups)", 0, 3, 1)
        improved_sleep = avg_before + 0.2 * reduction
        st.write(f"If coffee consumption is reduced {reduction} cup/day → average sleep increases to {improved_sleep:.2f} hours")

    with T4:
        st.subheader("Business Insights")
        st.markdown("""
        - **Coffee Retailer** → Product low caffeine for segment *high stress sleepers*  
        - **Wellness App** → Notification: “You've already had 3 cups, try cutting back to sleep better.”  
        - **Corporate HR** → Health campaign for office workers in countries with high coffee consumption  
        - **Policy Simulation** → If coffee consumption ↓1 cup/day → sleep quality ↑ in population X  
        """)

    