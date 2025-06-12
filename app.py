import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import seaborn as sns

client = OpenAI(api_key=("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"))

# Load dataset
def load_data():
    df = pd.read_csv("HR_FINPRO.csv")  # Path to your dataset
    return df

@st.cache_resource
def load_model():
    return joblib.load('rf_top_10.joblib')

model = load_model()
df = load_data()

# Mapping label asli dari data yang telah dienkode
label_mapping = {
    "Attrition": {0: "No", 1: "Yes"},
}

for column, mapping in label_mapping.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Sidebar
st.sidebar.title("HR Attrition Prediction")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Prediction", "Batch Prediction"])

st.title("HR Attrition Prediction")

# =============================== HOME ===============================
if menu == "Home":
    st.subheader("Dashboard Karyawan Berdasarkan Attrition")
    attrition_status = st.selectbox("Pilih Status Attrition:", df["Attrition"].unique())
    filtered_df = df[df["Attrition"] == attrition_status]

    if model is not None and hasattr(model, "feature_importances_"):
        st.write("### 🔍 Top 10 Fitur Berdasarkan Importance")
        try:
            feature_names = ['MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome', 
                             'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction', 
                             'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction']
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            plt.figure(figsize=(10, 5))
            sns.barplot(x="Importance", y="Fitur", data=feature_df, palette="viridis")
            plt.title("Top 10 Fitur Berdasarkan Importance")
            plt.tight_layout()
            st.pyplot(plt)

            st.write("### 📋 Tabel Top 10 Fitur")
            st.dataframe(feature_df.reset_index(drop=True))

        except Exception as e:
            st.error(f"Gagal memuat feature importance: {e}")
    else:
        st.warning("Model belum dimuat atau tidak memiliki atribut feature_importances_.")

    st.markdown("## 📊 Analisis Visual")

    if "OverTime" in df.columns:
        st.write("### 1. OverTime vs Attrition")
        overtime_attrition = df.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack().fillna(0)
        overtime_attrition.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Set2')
        plt.title("OverTime vs Attrition")
        plt.ylabel("Proportion")
        plt.xticks(rotation=0)
        st.pyplot(plt)
    else:
        st.warning("Kolom 'OverTime' tidak ditemukan di dataset.")

    if "MaritalStatus_Single" in df.columns and "MaritalStatus_Married" in df.columns:
        st.write("### 2. Attrition by Marital Status (Single vs Married)")
        df_marital = df[["Attrition", "MaritalStatus_Single", "MaritalStatus_Married"]].copy()
        df_marital = df_marital.melt(id_vars="Attrition", 
                                      value_vars=["MaritalStatus_Single", "MaritalStatus_Married"],
                                      var_name="Status", value_name="Value")
        df_marital = df_marital[df_marital["Value"] == 1]
        marital_summary = df_marital.groupby("Status")["Attrition"].value_counts(normalize=True).unstack().fillna(0)
        marital_summary.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Pastel1')
        plt.title("Attrition by Marital Status (Single vs Married)")
        plt.ylabel("Proportion")
        plt.xticks(rotation=0)
        st.pyplot(plt)
    elif "MaritalStatus" in df.columns:
        st.write("### 2. Attrition by Marital Status")
        marital_attrition = df.groupby("MaritalStatus")["Attrition"].value_counts(normalize=True).unstack().fillna(0)
        marital_attrition.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Pastel1')
        plt.title("Attrition by Marital Status")
        plt.xticks(rotation=0)
        st.pyplot(plt)
    else:
        st.warning("Kolom 'MaritalStatus' tidak ditemukan di dataset.")

# ============================ PREDICTION ============================
elif menu == "Prediction":
    if model is None:
        st.error("Model not loaded.")
    else:
        st.title('Employee Attrition Prediction')
        st.markdown("""<div style='background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
                            <span style='color:white'><b>This app predicts employee attrition based on key work-related factors 🚀. Enter the details and get an instant prediction!</b></span></div>""", unsafe_allow_html=True)

        st.markdown("### Enter Employee Details")

        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)

            with col1:
                MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
                MaritalStatus_Single = 1 if MaritalStatus == "Single" else 0
                JobLevelSatisfaction = st.selectbox("Job Level Satisfaction (1–4)", [1, 2, 3, 4])
                MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
                StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
                JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])

            with col2:
                EmployeeSatisfaction = st.selectbox("Employee Satisfaction (1–4)", [1, 2, 3, 4])
                DailyRate = st.number_input("Daily Rate (opsional)", min_value=0, max_value=1500, value=0)
                DistanceFromHome = st.number_input("Distance From Home (km)", 0, 100, 10)
                Age = st.number_input("Age", min_value=18, max_value=60, value=30)
                EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4])

            submit = st.form_submit_button("Predict")

        if submit:
            input_data = pd.DataFrame([[MaritalStatus_Single, JobLevelSatisfaction, MonthlyIncome,
                                        StockOptionLevel, JobInvolvement, EmployeeSatisfaction,
                                        DailyRate, DistanceFromHome, Age, EnvironmentSatisfaction]],
                                      columns=['MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome',
                                               'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction',
                                               'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction'])

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(input_data)[0]
                prob_resign = round(probas[1] * 100, 2)
                prob_stay = round(probas[0] * 100, 2)

                if prediction == 1:
                    st.info(f"📊 Peluang karyawan untuk **resign**: **{prob_resign}%**")
                else:
                    st.info(f"📊 Peluang karyawan untuk **bertahan**: **{prob_stay}%**")

            st.write("### 🤖 OpenAI Recommendation")

            prompt = f"""
            Based on the following employee data, provide a recommendation:

            Marital Status: {'Single' if MaritalStatus_Single == 1 else 'Married'}
            Job Level Satisfaction: {JobLevelSatisfaction}
            Monthly Income: {MonthlyIncome}
            Stock Option Level: {StockOptionLevel}
            Job Involvement: {JobInvolvement}
            Employee Satisfaction: {EmployeeSatisfaction}
            Daily Rate: {DailyRate}
            Distance from Home: {DistanceFromHome} km
            Age: {Age}
            Environment Satisfaction: {EnvironmentSatisfaction}

            Please provide a recommendation on how to improve retention for this employee.
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an HR assistant that helps improve employee retention."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                recommendation = response.choices[0].message.content.strip()
                st.write(f"### 💡 Recommendation: {recommendation}")
            except Exception as e:
                st.error(f"Failed to get recommendation from OpenAI: {e}")

# =============================== Batch Prediction ===============================
elif menu == "Batch Prediction":
    st.subheader("📂 Upload File Karyawan untuk Prediksi Massal")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Validasi apakah kolom sesuai dengan yang diharapkan oleh model
            expected_columns = [
                'MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome',
                'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction',
                'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction'
            ]

            if not all(col in batch_df.columns for col in expected_columns):
                st.error("⚠️ File CSV harus memiliki kolom berikut:\n" + ", ".join(expected_columns))
            else:
                # Prediksi
                predictions = model.predict(batch_df)
                probabilities = model.predict_proba(batch_df)

                batch_df['Prediksi'] = np.where(predictions == 1, 'Resign', 'Bertahan')
                batch_df['Peluang Resign (%)'] = (probabilities[:, 1] * 100).round(2)
                batch_df['Peluang Bertahan (%)'] = (probabilities[:, 0] * 100).round(2)

                st.success("✅ Prediksi berhasil dilakukan!")
                st.markdown("### 📊 Hasil Prediksi Karyawan:")
                st.dataframe(batch_df)

                # Download hasil
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Hasil Prediksi sebagai CSV",
                    data=csv,
                    file_name="hasil_prediksi_karyawan.csv",
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")

