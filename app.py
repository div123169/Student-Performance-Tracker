import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.predict import predict_single, get_performance_category
from src.utils import configure_plotting_style

# Set page configuration with a premium look
st.set_page_config(
    page_title="Student Performance Tracker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism look and feel)
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2C3E50;
        font-weight: 700;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 3rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .sidebar .sidebar-content {
        background-color: #1E293B;
    }
    .reportview-container .markdownTable {
        font-family: 'Inter', sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/isometric/512/student-male.png", width=100)
st.sidebar.title("Student Analytics Dashboard")
st.sidebar.markdown("---")

# Page selection
app_mode = st.sidebar.radio(
    "Navigate Project",
    ["🔮 Score Predictor", "📊 Dataset Insights", "⚙️ Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("About Internship Project")
st.sidebar.info(
    "**Title:** Student Performance Tracker using Machine Learning\n\n"
    "**Objective:** Build an intermediate ML system to predict student exam scores using demographic and study indicators."
)

# ----------------- PRE-LOAD METADATA -----------------
@st.cache_resource
def load_model_assets():
    try:
        metadata = joblib.load('models/best_model_metadata.joblib')
        comparison = pd.read_csv('models/model_comparison.csv')
        df_dataset = pd.read_csv('dataset/student_performance.csv')
        return metadata, comparison, df_dataset
    except Exception as e:
        return None, None, None

metadata, comparison, df_dataset = load_model_assets()

# Helper function to convert category to color
def get_category_color(cat: str) -> str:
    color_map = {
        'Excellent': '#10B981',        # Green
        'Good': '#3B82F6',             # Blue
        'Average': '#F59E0B',          # Orange
        'Needs Improvement': '#EF4444' # Red
    }
    return color_map.get(cat, '#6B7280')

# ----------------- PAGE: SCORE PREDICTOR -----------------
if app_mode == "🔮 Score Predictor":
    st.title("🎓 Student Performance Predictor")
    st.markdown("Enter the demographic and academic details below to predict a student's final exam score.")

    if metadata is None:
        st.warning("⚠️ Training assets not found. Please run the training pipeline first using `python src/train.py`.")
    else:
        best_model_name = metadata['best_model_name']
        
        # Two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Demographic Profile")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", min_value=15, max_value=22, value=17, step=1)
            parent_edu = st.selectbox(
                "Parent's Education Level", 
                ["None", "Primary Education", "5th to 9th Grade", "Secondary Education", "Higher Education"],
                index=4
            )
            internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
            family_support = st.selectbox("Family Educational Support", ["Yes", "No"])
            
        with col2:
            st.subheader("📚 Academic & Lifestyle Profile")
            attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=92.0, step=0.5)
            study_hours = st.slider("Weekly Study Hours", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
            previous_score = st.slider("Previous Score (0 - 100)", min_value=0, max_value=100, value=65, step=1)
            extra_activities = st.selectbox("Extra-Curricular Activities", ["Yes", "No"])
            sleep_hours = st.slider("Average Daily Sleep Hours", min_value=4.0, max_value=10.0, value=7.5, step=0.1)

        st.markdown("---")
        
        # Prediction button
        if st.button("🔮 Predict Student Performance", use_container_width=True):
            # Formulate the input dictionary
            student_profile = {
                'Gender': gender,
                'Age': age,
                'Study_Hours': study_hours,
                'Attendance': attendance,
                'Previous_Score': float(previous_score),
                'Parent_Education': parent_edu,
                'Internet_Access': internet,
                'Extra_Activities': extra_activities,
                'Family_Support': family_support,
                'Sleep_Hours': sleep_hours
            }
            
            with st.spinner("Processing features and running regression models..."):
                try:
                    result = predict_single(student_profile)
                    
                    score = result['predicted_score']
                    category = result['performance_category']
                    margin = result['prediction_margin']
                    lower, upper = result['confidence_interval']
                    
                    # Display results in columns
                    st.success("🎉 Prediction Calculated Successfully!")
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Predicted Final Score</div>
                                <div class="metric-val">{score}%</div>
                                <div style="color: #4B5563; font-weight: 500;">
                                    Range of certainty: <b>{lower}%</b> to <b>{upper}%</b> (±{margin}% error margin)
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                    with res_col2:
                        cat_color = get_category_color(category)
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Performance Classification</div>
                                <div class="metric-val" style="color: {cat_color};">{category}</div>
                                <div style="color: #4B5563; font-weight: 500;">
                                    Assessment based on structural metrics
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Contextual feedback and recommendation
                    st.subheader("💡 Actionable Recommendations")
                    recs = []
                    if attendance < 85.0:
                        recs.append("⚠️ **Low Attendance**: This student has an attendance of {:.1f}%. Increasing attendance above 90% is highly correlated with score improvements.".format(attendance))
                    if study_hours < 5.0:
                        recs.append("⚠️ **Limited Study Hours**: The student studies {:.1f} hours/week. Suggest increasing study time to at least 8-10 hours/week.".format(study_hours))
                    if sleep_hours < 7.0:
                        recs.append("⚠️ **Sleep Deprivation**: The student averages {:.1f} hours of sleep. Ensuring 7.5-8 hours of sleep enhances retention and cognitive focus.".format(sleep_hours))
                    if previous_score < 55.0:
                        recs.append("⚠️ **Core Subject Remediation**: Previous test scores are low ({:.0f}%). Enrolling in extra-paid tutor support or study circles is recommended.".format(previous_score))
                    
                    if len(recs) == 0:
                        st.write("✨ **Excellent Student Profile**: The student exhibits outstanding study practices. Keep up the high attendance and healthy sleep habits!")
                    else:
                        for rec in recs:
                            st.write(rec)
                            
                except Exception as e:
                    st.error(f"Prediction Pipeline Failed: {e}")

# ----------------- PAGE: DATASET INSIGHTS -----------------
elif app_mode == "📊 Dataset Insights":
    st.title("📊 Dataset Exploratory Analysis Insights")
    st.markdown("Browse raw summary parameters and distribution visualizations of the Student Performance Dataset.")
    
    if df_dataset is None:
        st.warning("⚠️ Cleaned dataset not found. Please verify the `dataset/student_performance.csv` exists.")
    else:
        # Show metrics cards of dataset statistics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Records", f"{len(df_dataset)}")
        mcol2.metric("Mean Final Score", f"{df_dataset['Final_Score'].mean():.1f}%")
        mcol3.metric("Average Attendance", f"{df_dataset['Attendance'].mean():.1f}%")
        mcol4.metric("Avg. Study Hours", f"{df_dataset['Study_Hours'].mean():.1f} hrs/wk")
        
        tab1, tab2, tab3 = st.tabs(["📋 Summary Data", "📈 Feature Distributions", "🔗 Feature Correlations"])
        
        with tab1:
            st.subheader("Dataset Structure & Statistics")
            st.dataframe(df_dataset.head(10), use_container_width=True)
            
            st.markdown("#### Feature Explanations")
            st.markdown(
                "- **Gender**: Male / Female\n"
                "- **Age**: Age of student (15-22 years)\n"
                "- **Study Hours**: Continuous weekly hours of private study\n"
                "- **Attendance**: School class attendance percentage (calculated from absences)\n"
                "- **Previous Score**: First period examination grade (0-100)\n"
                "- **Parent Education**: Maximum education level attained by parents\n"
                "- **Internet Access**: Access to home internet (Yes/No)\n"
                "- **Extra Activities / Family Support**: Socio-educational indicators"
            )
            
        with tab2:
            st.subheader("Distribution Visualizations")
            
            # Sub-columns for selectors
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                plot_col = st.selectbox(
                    "Select Feature to Plot Distribution", 
                    ['Final_Score', 'Attendance', 'Study_Hours', 'Sleep_Hours', 'Previous_Score']
                )
            with sel_col2:
                bins = st.slider("Number of Histogram Bins", min_value=5, max_value=30, value=15)
                
            configure_plotting_style()
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.histplot(df_dataset[plot_col].dropna(), kde=True, color='#4E79A7', bins=bins, ax=ax)
            ax.set_title(f'Distribution of {plot_col}')
            ax.set_xlabel(plot_col)
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            st.markdown("#### Categorical Categorization Plots")
            fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            sns.boxplot(data=df_dataset, x='Gender', y='Final_Score', ax=axes[0], palette='pastel')
            axes[0].set_title("Final Score by Gender")
            axes[0].set_ylabel("Final Score (%)")
            
            sns.boxplot(data=df_dataset, x='Family_Support', y='Final_Score', ax=axes[1], palette='Set2')
            axes[1].set_title("Final Score by Family Support")
            axes[1].set_ylabel("Final Score (%)")
            st.pyplot(fig2)
            
        with tab3:
            st.subheader("Statistical Correlation Matrix")
            configure_plotting_style()
            
            numeric_cols = ['Age', 'Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours', 'Final_Score']
            fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
            sns.heatmap(df_dataset[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
            ax_corr.set_title("Correlation heatmap of student indicators")
            st.pyplot(fig_corr)
            st.write("🔗 **Key Observation**: *Previous Score* and *Attendance* exhibit the highest positive correlations with the target variable *Final Score*.")

# ----------------- PAGE: MODEL PERFORMANCE -----------------
elif app_mode == "⚙️ Model Performance":
    st.title("⚙️ Regression Model Performance & Metrics")
    st.markdown("Compare the performance of all machine learning models tested in the pipeline.")
    
    if comparison is None:
        st.warning("⚠️ Model comparison data not found. Please run the training script first.")
    else:
        # Display comparison table
        st.subheader("Regression Models Leaderboard")
        st.markdown("Ordered by $R^2$ Score on the test partition (80:20 split):")
        
        # Display styled table
        st.dataframe(
            comparison.style.highlight_max(subset=['R2'], color='#D1FAE5')
                             .highlight_min(subset=['RMSE', 'MAE'], color='#FEE2E2'),
            use_container_width=True
        )
        
        # Explanation of Metrics
        st.markdown(
            "##### Metric Definitions:\n"
            "- **$R^2$ Score**: Proportion of variance in final scores explained by the indicators (closer to 1.0 is better).\n"
            "- **MAE (Mean Absolute Error)**: Average absolute deviance between predicted score and actual score (in score percentage units).\n"
            "- **RMSE (Root Mean Squared Error)**: Standard deviation of predictions; penalizes larger errors more heavily (closer to 0 is better)."
        )
        
        st.markdown("---")
        st.subheader("Model Visualizations")
        
        m_tab1, m_tab2 = st.tabs(["🎯 Actual vs. Predicted Fit", "🌲 Feature Importances"])
        
        with m_tab1:
            if os.path.exists('notebooks/plots/actual_vs_predicted.png'):
                st.image('notebooks/plots/actual_vs_predicted.png', caption="Actual vs. Predicted Scores Scatterplot", use_container_width=True)
            else:
                st.info("Actual vs Predicted plot image not found in notebooks/plots/ directory.")
                
            if os.path.exists('notebooks/plots/residuals_plot.png'):
                st.image('notebooks/plots/residuals_plot.png', caption="Residuals Distribution Homoscedasticity check", use_container_width=True)
            else:
                st.info("Residuals plot image not found in notebooks/plots/ directory.")
                
        with m_tab2:
            if os.path.exists('notebooks/plots/feature_importance.png'):
                st.image('notebooks/plots/feature_importance.png', caption="Learned Feature Importances from the best model", use_container_width=True)
            else:
                st.info("Feature importance plot image not found in notebooks/plots/ directory.")
