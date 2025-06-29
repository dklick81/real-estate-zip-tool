# app.py
import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fpdf import FPDF
from io import BytesIO
import datetime
from streamlit_folium import st_folium
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import re
from pathlib import Path
import os
import boto3
from io import StringIO

@st.cache_data
def load_data():
    aws = st.secrets["aws"]
    s3 = boto3.client("s3", aws_access_key_id=aws["aws_access_key_id"], aws_secret_access_key=aws["aws_secret_access_key"])
    obj = s3.get_object(Bucket=aws["staging_bucket"], Key="real_zip_scores.csv")
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype={"zip_code": str})

# Load data and initialize session state
df = load_data()
for step in ['step1', 'step2', 'step3']:
    if step not in st.session_state:
        st.session_state[step] = False

st.title("Real Estate Neighborhood Ranking Tool")

def render_pipeline_progress():
    steps = {
        "Step 1: Filter ZIP Codes": st.session_state.step1,
        "Step 2: Clean & Score": st.session_state.step2,
        "Step 3: View Results": st.session_state.step3
    }
    st.markdown("### 🧭 Pipeline Progress")
    for label, done in steps.items():
        status = "✅" if done else "⏳"
        st.markdown(f"{status} **{label}**")

render_pipeline_progress()

# --- Sidebar Inputs ---
st.sidebar.title("Search Options")
user_zip = st.sidebar.text_input("Enter the desired central ZIP code:", "44102")
#Input validation (verifying 5 digit ZIP code is entered)
if not re.fullmatch(r"\d{5}", user_zip):
    st.sidebar.error("⚠️ ZIP code must be exactly 5 digits.")
    st.stop()
radius = st.sidebar.slider("Search radius (miles):", 1, 30, 10)

st.sidebar.title("Ranking Preferences")
afford = st.sidebar.slider("Affordability", 0, 100, 17, key='affordability')
taxes = st.sidebar.slider("Property Tax", 0, 100, 17, key='property_tax')
school = st.sidebar.slider("School Quality", 0, 100, 17, key='school_quality')
crime = st.sidebar.slider("Low Crime (Higher = Safer)", 0, 100, 17, key='crime_rate')
walk = st.sidebar.slider("Walkability", 0, 100, 16, key='walkability')
commute = st.sidebar.slider("Commute Time", 0, 100, 16, key='commute_time')

#help section
with st.sidebar.expander("❓ Help", expanded=False):
    st.markdown(
        """
        **Search Radius**  
        Controls how far (in miles) from your center ZIP to include neighborhoods in the analysis.  
        
        **Ranking Weights**  
        Slide the bars to adjust each metric’s importance; the final score is a weighted sum:
        ```python
        score = affordability*w1 + (1-tax)*w2 + school_quality*w3 + …
        ```
        
        **Analysis Methods**  
        - **Histogram**: Distribution of your chosen metric across the top ZIPs.  
        - **Boxplot**: Quartiles, median, and outliers for that metric.  
        - **Radar**: Multi-axis view comparing all metrics for one ZIP.  
        
        **Map Markers**  
        Click a circle to see full metric breakdown and overall score. Red pin marks your center ZIP.  
        
        **Downloads**  
        Use CSV/PDF buttons to export your filtered data or rankings for offline review.  
        For more details, see our [GitHub Wiki](https://github.com/dklick81/real-estate-zip-tool/wiki).
        """
    )

if st.sidebar.button("🚀 Run Full Analysis"):
    # Step 1
    user_row = df[df['zip_code'] == user_zip]
    if not user_row.empty:
        user_location = (user_row.iloc[0]['latitude'], user_row.iloc[0]['longitude'])
        st.session_state.user_location = user_location

        df['distance'] = df.apply(lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).miles, axis=1)
        df = df[df['distance'] <= radius]

        if not df.empty:
            st.session_state.step1 = True
            st.session_state.df = df

            # Step 2
            if st.session_state.step1:
                df = st.session_state.df.copy()
                norm_cols = ["affordability", "property_tax", "crime_rate", "school_quality", "walkability", "commute_time"]
                scaler = MinMaxScaler()
                df[norm_cols] = scaler.fit_transform(df[norm_cols])

                total = sum([afford, taxes, school, crime, walk, commute])
                weights = {
                    'affordability': afford / total,
                    'property_tax': taxes / total,
		    'school_quality': school / total,
                    'crime_rate': crime / total,
                    'walkability': walk / total,
                    'commute_time': commute / total
                }

                df["score"] = (
                    df["affordability"] * weights['affordability'] +
                    (1 - df["property_tax"]) * weights['property_tax'] +
		    df["school_quality"] * weights['school_quality'] +
                    (1 - df["crime_rate"]) * weights['crime_rate'] +
                    df["walkability"] * weights['walkability'] +
                    (1 - df["commute_time"]) * weights['commute_time']
                )

                st.session_state.step2 = True
                st.session_state.df = df
                st.session_state.top_zips = df.sort_values("score", ascending=False).head(10)[["zip_code", "score"]]
                st.session_state.weights = weights
                st.session_state.step3 = True


# --- Step 1: Filter by ZIP and Radius ---
st.header("Step 1: Select ZIP and Radius")
if st.button("✅ Run Step 1: Filter ZIP Codes"):
    user_row = df[df['zip_code'] == user_zip]
    if user_row.empty:
        st.error("ZIP code not found in dataset.")
        st.stop()

    user_location = (user_row.iloc[0]['latitude'], user_row.iloc[0]['longitude'])
    st.session_state.user_location = user_location

    df['distance'] = df.apply(lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).miles, axis=1)
    df = df[df['distance'] <= radius]

    if df.empty:
        st.warning("No ZIP codes found within that radius.")
        st.stop()

    st.session_state.step1 = True
    st.session_state.df = df
    st.success("ZIP filtering complete.")

# --- Step 2: Clean and Score ---
if st.session_state.step1:
    st.header("Step 2: Apply Cleaning and Preferences")

    st.sidebar.title("Data Cleaning Options")
    outlier_removal = st.sidebar.checkbox("Simulate outlier removal (affordability < 0.3)", value=False, key="outliers")
    re_normalize = st.sidebar.checkbox("Re-normalize scoring columns after filtering", value=False, key="normalize")

    if st.button("✅ Run Step 2: Clean Data and Score"):
        df = st.session_state.df.copy()

        
        norm_cols = ["affordability", "property_tax", "crime_rate", "school_quality", "walkability", "commute_time"]
        scaler = MinMaxScaler()
        df[norm_cols] = scaler.fit_transform(df[norm_cols])

        total = sum([afford, taxes, school, crime, walk, commute])
        weights = {
            'affordability': afford / total,
            'property_tax': taxes / total,
	    'school_quality': school / total,
            'crime_rate': crime / total,
            'walkability': walk / total,
            'commute_time': commute / total
        }

        df["score"] = (
            df["affordability"] * weights['affordability'] +
            (1 - df["property_tax"]) * weights['property_tax'] +
	    df["school_quality"] * weights['school_quality'] +
            (1 - df["crime_rate"]) * weights['crime_rate'] +
            df["walkability"] * weights['walkability'] +
            (1 - df["commute_time"]) * weights['commute_time']
        )

        st.session_state.step2 = True
        st.session_state.df = df
        st.session_state.top_zips = df.sort_values("score", ascending=False).head(10)[["zip_code", "score"]]
        st.success("Scoring complete.")

# --- Step 3: Show Results ---
if st.session_state.step1 and st.session_state.step2:
    if st.button("✅ Run Step 3: Generate Output"):
        st.session_state.step3 = True
    
    # --- Dataset View ---
    top_zips = st.session_state.top_zips
    top_codes = top_zips["zip_code"].tolist()
    viz_df    = df[df["zip_code"].isin(top_codes)]
    st.sidebar.title("Dataset View")
    view_option = st.sidebar.radio(
        "Choose which dataset to display:",
        ["Raw Data", "Filtered Data", "Top Ranked ZIPs"]
    )

    if st.session_state.step3:
        df = st.session_state.df
        user_location = st.session_state.user_location
        top_zips = st.session_state.top_zips

        st.subheader("Top ZIP Codes")
        top_zips_clean = top_zips.reset_index(drop=True)
        st.dataframe(top_zips_clean,hide_index=True)
        st.download_button(
            label="📥 Download Top ZIPs as CSV",
            data=top_zips.to_csv(index=False),
            file_name='top_zip_scores.csv',
            mime='text/csv'
        )

        def generate_pdf_report(zip_code, radius, weights, top_df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Top ZIP Code Rankings", ln=True, align="C")
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(10)

            pdf.cell(200, 10, txt=f"Search Center ZIP Code: {user_zip}", ln=True)
            pdf.cell(200, 10, txt=f"Search Radius: {radius} miles", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", "B", 10)
            pdf.cell(200, 10, txt="Ranking Weights:", ln=True)
            pdf.set_font("Arial", size=10)
            for k, v in weights.items():
                pdf.cell(200, 8, txt=f"{k.replace('_', ' ').title()}: {round(v*100)}%", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", "B", 10)
            pdf.cell(200, 10, txt="Top ZIPs:", ln=True)
            pdf.set_font("Arial", size=10)

            for index, row in top_df.iterrows():
                pdf.cell(200, 8, txt=f"{row['zip_code']} - Score: {round(row['score'], 2)}", ln=True)

            buffer = BytesIO()
            pdf_output = pdf.output(dest='S').encode('latin1')
            buffer.write(pdf_output)
            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf_report(user_zip, radius, st.session_state.get('weights', {}), top_zips)
        st.download_button(
            label="📄 Download Top ZIPs as PDF",
            data=pdf_buffer,
            file_name="top_zip_report.pdf",
            mime="application/pdf"
        )

        with st.container():
            st.subheader("Dataset Preview")
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=df.to_csv(index=False),
                file_name='filtered_zip_data.csv',
                mime='text/csv'
            )
            if view_option == "Raw Data":
                st.dataframe(load_data())
            elif view_option == "Filtered Data":
                df_clean = df.reset_index(drop=True)
                st.dataframe(df_clean.style.hide(0))
            elif view_option == "Top Ranked ZIPs":
                if "top_zips" in st.session_state:
                    st.dataframe(st.session_state.top_zips)
                else:
                    st.warning("Please run Step 2 first to generate the top ZIP rankings.")

        st.subheader("Map and Data Exploration")

        col1, col2 = st.columns([1, 1])
        with col1:
            m = folium.Map(location=user_location, zoom_start=11)
            folium.Marker(user_location, tooltip="You are here", icon=folium.Icon(color="red")).add_to(m)
            for _, row in viz_df.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=7,
                    popup=(
                        f"ZIP: {row['zip_code']}<br>"
 #                       f"Score: {round(row['score'], 2)}<br>"
                        f"Affordability: {round(row['affordability'], 2)}<br>"
                        f"Property Tax: {round(row['property_tax'], 2)}<br>"
			f"Schools: {round(row['school_quality'], 2)}<br>"
                        f"Crime: {round(row['crime_rate'], 2)}<br>"
                        f"Walkability: {round(row['walkability'], 2)}<br>"
                        f"Commute: {round(row['commute_time'], 2)}"
                    ),
                    color="blue",
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
            st_folium(m, width=350, height=350)

        with col2:
            chart_type = st.selectbox("Choose chart type:", ["Histogram", "Boxplot", "Radar"])
            chart_var = st.selectbox("Select variable to visualize:", [
                "affordability", "property_tax", "crime_rate", "school_quality", "walkability", "commute_time"])

            if chart_type in ["Histogram", "Boxplot"]:
                st.markdown(f"#### {chart_type} for {chart_var}")
                if chart_type == "Histogram":
                    st.bar_chart(viz_df[chart_var])
                elif chart_type == "Boxplot":
                    fig, ax = plt.subplots()
                    sns.boxplot(y=viz_df[chart_var], ax=ax)
                    st.pyplot(fig)

            elif chart_type == "Radar":
                st.markdown("#### Radar Chart for a Single ZIP Code")
                selected_zip = st.selectbox("Choose a ZIP code to display:", df['zip_code'].unique())
                radar_row = df[df['zip_code'] == selected_zip].iloc[0]
                categories = ["affordability", "property_tax", "crime_rate", "school_quality", "walkability", "commute_time"]
                values = [radar_row[var] if var not in ["crime_rate", "commute_time"] else 1 - radar_row[var] for var in categories]
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.plot(angles, values, linewidth=2, linestyle='solid')
                ax.fill(angles, values, alpha=0.25)
                ax.set_yticklabels([])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_title(f"ZIP Code: {selected_zip}", y=1.1)
                st.pyplot(fig)

#reset
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Reset Analysis")
if st.sidebar.button("Reset Pipeline"):
    for step in ['step1', 'step2', 'step3']:
        st.session_state[step] = False
    st.rerun()