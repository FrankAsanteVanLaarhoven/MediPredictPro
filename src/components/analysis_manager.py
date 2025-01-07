import streamlit as st
import plotly.express as px
import pandas as pd

def render_analysis(df: pd.DataFrame):
    """Render analysis interface"""
    st.title("Medicine Data Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Overview", "Review Analysis", "Manufacturer Analysis", "Side Effects"]
    )
    
    if analysis_type == "Overview":
        render_overview(df)
    elif analysis_type == "Review Analysis":
        render_review_analysis(df)
    elif analysis_type == "Manufacturer Analysis":
        render_manufacturer_analysis(df)
    elif analysis_type == "Side Effects":
        render_side_effects_analysis(df)

def render_overview(df: pd.DataFrame):
    st.subheader("Data Overview")
    st.dataframe(df.head())
    st.write("Dataset Info:")
    st.write(df.describe())

def render_review_analysis(df: pd.DataFrame):
    st.subheader("Review Score Analysis")
    fig = px.violin(
        df,
        y=["Excellent Review %", "Average Review %", "Poor Review %"],
        box=True,
        title="Review Score Distribution by Type"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_manufacturer_analysis(df: pd.DataFrame):
    st.subheader("Manufacturer Analysis")
    top_mfg = (df.groupby('Manufacturer')
              .agg({
                  'Excellent Review %': 'mean',
                  'Medicine': 'count'
              })
              .sort_values('Excellent Review %', ascending=False)
              .head(10))
    
    fig = px.bar(
        top_mfg,
        y='Excellent Review %',
        title="Top Manufacturers by Review Score",
        text='Medicine'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_side_effects_analysis(df: pd.DataFrame):
    st.subheader("Side Effects Analysis")
    side_effects = (df['Side_effects']
                   .str.split()
                   .explode()
                   .value_counts()
                   .head(15))
    
    fig = px.bar(
        x=side_effects.index,
        y=side_effects.values,
        title="Most Common Side Effects"
    )
    st.plotly_chart(fig, use_container_width=True)