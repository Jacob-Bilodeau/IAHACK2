import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Analytics Dashboard", layout="wide")


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0068c9;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("RAPPORT DE PERFORMANCE ANALYTIQUE")
st.markdown("---")

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "resultats.json")

try:
    with open(json_path, "r") as f:
        data = json.load(f)


    metrics_pct = {k: v * 100 for k, v in data.items()}


    cols = st.columns(5)
    labels = ["ACCURACY", "PRECISION", "RECALL", "F1 SCORE", "DIAG MEAN"]
    keys = ["accuracy", "precision", "recall", "f1_score", "diag_mean"]

    for col, label, key in zip(cols, labels, keys):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <p style='color: #555; font-size: 0.9em; font-weight: bold; margin-bottom: 5px;'>{label}</p>
                    <h2 style='color: #0068c9; margin: 0;'>{metrics_pct[key]:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    
    col_left, col_right = st.columns([2, 1])

    with col_left:

        df_plot = pd.DataFrame({
            'Metrique': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score (%)': [metrics_pct['accuracy'], metrics_pct['precision'], metrics_pct['recall'], metrics_pct['f1_score']]
        })

       
        fig = px.bar(df_plot, 
                     x='Score (%)', 
                     y='Metrique', 
                     orientation='h',
                     text='Score (%)',
                     color='Score (%)',
                     color_continuous_scale='Blues',
                     range_x=[0, 105])

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.8)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=14),
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(autorange="reversed"), 
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Analyse de Distribution")
     
        fig_gauge = px.pie(values=[data['f1_score'], 1-data['f1_score']], 
                           names=['Performance', 'Reste'],
                           hole=0.7,
                           color_discrete_sequence=['#0068c9', '#e0e0e0'])
        fig_gauge.update_traces(textinfo='none')
        fig_gauge.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"<h3 style='text-align: center; margin-top: -160px;'>{metrics_pct['f1_score']:.1f}%</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: grey;'>F1 Global</p>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Fichier de résultats introuvable.")