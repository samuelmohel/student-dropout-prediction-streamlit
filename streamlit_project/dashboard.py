import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/HP/Desktop/transformed_data2.csv')
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")

# Example filters based on available columns
gender_options = df['Gender'].unique().tolist()
scholarship_options = df['Scholarship holder'].unique().tolist()
displaced_options = df['Displaced'].unique().tolist()
debtor_options = df['Debtor'].unique().tolist()

selected_gender = st.sidebar.multiselect("Select Gender", gender_options, default=gender_options)
selected_scholarship = st.sidebar.multiselect("Select Scholarship Status", scholarship_options, default=scholarship_options)
selected_displaced = st.sidebar.multiselect("Select Displaced Status", displaced_options, default=displaced_options)
selected_debtor = st.sidebar.multiselect("Select Debtor Status", debtor_options, default=debtor_options)

# Apply Filters
filtered_df = df[
    (df['Gender'].isin(selected_gender)) &
    (df['Scholarship holder'].isin(selected_scholarship)) &
    (df['Displaced'].isin(selected_displaced)) &
    (df['Debtor'].isin(selected_debtor))
]

# Main Dashboard Title
st.title("🎓 Student Performance Dashboard")

# Layout: Use columns to arrange multiple plots
col1, col2 = st.columns(2)

# 1. Scatter Plot: Admission Grade vs Academic Performance Ratio
with col1:
    st.subheader("Admission Grade vs Academic Performance Ratio")
    fig_scatter = px.scatter(
        filtered_df, 
        x='Admission grade', 
        y='Academic Performance Ratio',
        color='Gender',
        title="Admission Grade vs Academic Performance Ratio",
        hover_data=['Age at enrollment', 'Scholarship holder']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# 2. Histogram: Age at Enrollment
with col2:
    st.subheader("Distribution of Age at Enrollment")
    fig_hist = px.histogram(
        filtered_df, 
        x='Age at enrollment', 
        nbins=20,
        title="Age at Enrollment Distribution",
        color='Gender',
        marginal="box",
        hover_data=filtered_df.columns
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Additional Layout for More Visualizations
col3, col4 = st.columns(2)

# 3. Bar Chart: Count of Students by Gender
with col3:
    st.subheader("Count of Students by Gender")
    fig_bar = px.bar(
        filtered_df['Gender'].value_counts().reset_index(),
        x='index',
        y='Gender',
        labels={'index': 'Gender', 'Gender': 'Count'},
        title="Number of Students by Gender",
        color='index'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 4. Box Plot: Admission Grade by Scholarship Holder
with col4:
    st.subheader("Admission Grade by Scholarship Status")
    fig_box = px.box(
        filtered_df, 
        x='Scholarship holder', 
        y='Admission grade',
        title="Admission Grade Distribution by Scholarship Status",
        color='Scholarship holder'
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Full-Width Section for Correlation Heatmap
st.subheader("Correlation Heatmap of Numerical Features")
# Select numerical columns for correlation
numerical_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
corr_matrix = filtered_df[numerical_cols].corr()

# Create a Seaborn heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Additional Insights
st.subheader("Basic Statistics of Filtered Data")
st.write(filtered_df.describe())

# Optional: Download Filtered Data
st.subheader("Download Filtered Data")
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)

st.download_button(
    label="📥 Download data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
)




st.sidebar.header("Dynamic Plot Options")
x_axis = st.sidebar.selectbox("Select X-axis for Scatter Plot", options=numerical_cols, index=numerical_cols.index('Admission grade'))
y_axis = st.sidebar.selectbox("Select Y-axis for Scatter Plot", options=numerical_cols, index=numerical_cols.index('Academic Performance Ratio'))

fig_dynamic_scatter = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    title=f"{x_axis} vs {y_axis}",
    color='Gender',
    hover_data=['Age at enrollment', 'Scholarship holder']
)
st.plotly_chart(fig_dynamic_scatter, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Distribution", "Statistics"])

with tab1:
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.write(filtered_df.describe())


st.subheader("Filtered Data")
st.dataframe(filtered_df)


from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for missing values
st.write("Missing values in the dataset:")
st.write(df.isnull().sum())

# Option 1: Drop rows with missing values (you can also drop columns if needed)
df_clean = df.dropna()

# Option 2: Fill NaN values with the mean of each column (imputation)
# df_clean = df.fillna(df.mean())

# 2. Perform Standard Scaling (PCA requires scaled data)
features = df_clean.select_dtypes(include=[np.number])  # Select only numeric columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. Perform PCA
pca = PCA(n_components=2)  # You can change n_components as needed
pca_result = pca.fit_transform(scaled_features)

# Add PCA results back to DataFrame
df_clean['PCA1'] = pca_result[:, 0]
df_clean['PCA2'] = pca_result[:, 1]

# 4. Visualize the PCA results using Plotly in Streamlit
import plotly.express as px

fig = px.scatter(df_clean, x='PCA1', y='PCA2', color='Gender',
                 title="PCA Scatter Plot")
st.plotly_chart(fig)


# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Plotly")