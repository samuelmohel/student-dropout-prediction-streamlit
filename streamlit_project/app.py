import streamlit as st

# Save this in a file called `app.py`
import pandas as pd
import plotly.express as px

# Load your dataset
df = pd.read_csv("C:/Users/HP/Desktop/transformed_data2.csv")  # Replace with your dataset file

st.title("Student Dropout Prediction Dashboard")

# Plot 2: Interactive histogram
st.subheader("Admission Grade Histogram")
fig = px.histogram(df, x='Admission grade', color='Gender', title="Admission Grade Distribution by Gender")
st.plotly_chart(fig)

# Plot 3: Interactive box plot
st.subheader("Admission Grade by Gender")
fig = px.box(df, x='Gender', y='Admission grade', title="Admission Grade by Gender")
st.plotly_chart(fig)

# Set up the Streamlit app title
st.title("Admission Grade vs Academic Performance Ratio")

# Create a scatter plot
fig = px.scatter(df, x='Admission grade', y='Academic Performance Ratio',
                 title="Scatter Plot: Admission Grade vs Academic Performance Ratio")

# Display the scatter plot in Streamlit
st.plotly_chart(fig)

# Optionally display some basic statistics or other insights
st.write(df[['Admission grade', 'Academic Performance Ratio']].describe())


