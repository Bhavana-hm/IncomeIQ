
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


# ================================================
# PAGE CONFIGURATION
# ================================================

st.set_page_config(
    page_title="IncomeIQ | Income Prediction",
    page_icon="💰",
    layout="wide"
)


# ================================================
# CUSTOM WEBSITE STYLING
# ================================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label {
    color: #CBD5E1 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stMetric"] {
    background-color: #172554;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ================================================
# LOAD MACHINE LEARNING MODEL
# ================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/random_forest_model.pkl")

    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )

    return model, feature_columns


rf_model, feature_columns = load_model()


# ================================================
# LOAD DATASET
# ================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "datase/adult.csv",
        na_values="?"
    )

    return df


df = load_data()


# ================================================
# SIDEBAR NAVIGATION
# ================================================

st.sidebar.title("💰 IncomeIQ")

st.sidebar.caption("Smart Income Prediction")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Income Prediction",
        "Model Insights",
        "Model Performance",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Powered by Random Forest Machine Learning"
)


# ================================================
# PAGE 1: HOME
# ================================================

if page == "Home":

    # Hero Section
    st.markdown("""
        <h1 style="color:white; font-size:42px; margin-bottom:10px;">
            Welcome to IncomeIQ 💰
        </h1>

        <p style="color:#DBEAFE; font-size:19px;">
            Smart Income Prediction Powered by Artificial Intelligence
        </p>

        <p style="color:#BFDBFE; font-size:15px;">
            Discover income patterns and predict annual income
            using Random Forest Machine Learning.
        </p>
    """, unsafe_allow_html=True)
    


    # Dashboard Statistics

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Accuracy", "84.96%", "Random Forest")

    with col2:
        st.metric("Training Records", "26,029", "80% of dataset")

    with col3:
        st.metric("Testing Records", "6,508", "20% of dataset")

    st.divider()


    # About the Project

    st.subheader("🚀 About IncomeIQ")

    st.write(
        "IncomeIQ is an interactive machine learning application "
        "that predicts whether an individual's annual income is "
        "above or below $50K based on personal and employment details."
    )

    st.divider()


    # Feature Cards

    st.subheader("✨ Explore Our Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div style="
            background:#172554;
            padding:24px;
            border-radius:15px;
            border:1px solid #2563EB;
            min-height:155px;
        ">
            <h3>💰 Income Prediction</h3>
            <p>
                Predict annual income using personal and employment
                information with our trained Random Forest model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:#172554;
            padding:24px;
            border-radius:15px;
            border:1px solid #2563EB;
            min-height:155px;
            margin-top:15px;
        ">
            <h3>📈 Model Performance</h3>
            <p>
                Evaluate model performance through confusion matrix,
                precision, recall and F1-score.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div style="
            background:#172554;
            padding:24px;
            border-radius:15px;
            border:1px solid #2563EB;
            min-height:155px;
        ">
            <h3>📊 Model Insights</h3>
            <p>
                Explore income distribution and understand the
                most important features used by the model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:#172554;
            padding:24px;
            border-radius:15px;
            border:1px solid #2563EB;
            min-height:155px;
            margin-top:15px;
        ">
            <h3>🤖 Random Forest AI</h3>
            <p>
                Discover how an ensemble of decision trees
                helps generate income predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info(
        "Use the navigation menu on the left to explore IncomeIQ."
    )

    st.caption(
        "IncomeIQ | Smart Income Prediction Using Machine Learning"
    )


# ================================================
# PAGE 2: INCOME PREDICTION
# ================================================

elif page == "Income Prediction":

    st.title("💰 Income Prediction")

    st.subheader("Enter Personal and Employment Details")

    st.write(
        "Fill in the information below to predict the annual income category."
    )

    st.divider()

    # Prediction Form

    with st.form("income_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=17,
                max_value=100,
                value=30
            )

            workclass = st.selectbox(
                "Workclass",
                sorted(df["workclass"].dropna().unique().tolist())
            )

            education = st.selectbox(
                "Education",
                sorted(df["education"].dropna().unique().tolist())
            )

            marital_status = st.selectbox(
                "Marital Status",
                sorted(df["marital.status"].dropna().unique().tolist())
            )

            occupation = st.selectbox(
                "Occupation",
                sorted(df["occupation"].dropna().unique().tolist())
            )

            relationship = st.selectbox(
                "Relationship",
                sorted(df["relationship"].dropna().unique().tolist())
            )

            sex = st.selectbox(
                "Sex",
                sorted(df["sex"].dropna().unique().tolist())
            )

        with col2:

            fnlwgt = st.number_input(
                "Final Weight (fnlwgt)",
                min_value=1,
                value=189778
            )

            education_num = st.number_input(
                "Education Number",
                min_value=1,
                max_value=16,
                value=10
            )

            race = st.selectbox(
                "Race",
                sorted(df["race"].dropna().unique().tolist())
            )

            capital_gain = st.number_input(
                "Capital Gain",
                min_value=0,
                value=0
            )

            capital_loss = st.number_input(
                "Capital Loss",
                min_value=0,
                value=0
            )

            hours_per_week = st.number_input(
                "Hours Worked Per Week",
                min_value=1,
                max_value=100,
                value=40
            )

            native_country = st.selectbox(
                "Native Country",
                sorted(df["native.country"].dropna().unique().tolist())
            )

        submitted = st.form_submit_button(
            "🔍 Predict Income",
            use_container_width=True
        )


    # Prediction Logic

    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education.num": education_num,
            "marital.status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital.gain": capital_gain,
            "capital.loss": capital_loss,
            "hours.per.week": hours_per_week,
            "native.country": native_country
        }])

        input_encoded = pd.get_dummies(
            input_data,
            drop_first=True,
            dtype=int
        )

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = rf_model.predict(input_encoded)[0]

        st.divider()

        st.header("📊 Income Prediction Result")

        if prediction == ">50K":

            st.success(
                "Predicted Income: Above $50K per year"
            )

        else:

            st.info(
                "Predicted Income: $50K or below per year"
            )

        st.caption(
            "Prediction generated using your trained Random Forest model."
        )


# ================================================
# PAGE 3: MODEL INSIGHTS
# ================================================

elif page == "Model Insights":

    st.title("📊 Machine Learning Insights")

    st.write(
        "Explore the income distribution and important features "
        "used by the Random Forest model."
    )

    st.divider()

    # Income Distribution

    st.subheader("Income Distribution")

    income_counts = df["income"].value_counts().reset_index()

    income_counts.columns = [
        "Income Category",
        "Number of People"
    ]

    fig1 = px.pie(
        income_counts,
        names="Income Category",
        values="Number of People",
        hole=0.5,
        title="Income Distribution in Dataset"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.divider()

    # Feature Importance

    st.subheader("Important Features")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features",
        text="Importance"
    )

    fig2.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ================================================
# PAGE 4: MODEL PERFORMANCE
# ================================================

elif page == "Model Performance":

    st.title("📈 Model Performance Evaluation")

    st.write(
        "Evaluate the performance of the trained Random Forest model "
        "using standard classification metrics."
    )

    st.divider()

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    confusion_data = pd.DataFrame(
        [
            [4576, 364],
            [615, 953]
        ],
        index=["Actual ≤50K", "Actual >50K"],
        columns=["Predicted ≤50K", "Predicted >50K"]
    )

    fig3 = px.imshow(
        confusion_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Random Forest Confusion Matrix"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.divider()

    # Classification Report

    st.subheader("Classification Report")

    report_data = pd.DataFrame({

        "Class": [
            "≤50K",
            ">50K",
            "Macro Average",
            "Weighted Average"
        ],

        "Precision": [
            0.8815,
            0.7236,
            0.8026,
            0.8435
        ],

        "Recall": [
            0.9263,
            0.6078,
            0.7670,
            0.8496
        ],

        "F1-Score": [
            0.9034,
            0.6607,
            0.7820,
            0.8449
        ],

        "Support": [
            4940,
            1568,
            6508,
            6508
        ]

    })

    st.dataframe(
        report_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Evaluation results from the trained Random Forest model."
    )


# ================================================
# PAGE 5: ABOUT
# ================================================

elif page == "About":

    st.title("ℹ️ About IncomeIQ")

    st.subheader(
        "Smart Income Prediction Using Random Forest"
    )

    st.write(
        "IncomeIQ is a machine learning project designed to predict "
        "whether a person's annual income is above or below $50K."
    )

    st.divider()

    # Project Information

    st.header("Project Information")

    st.markdown("""
    **Project Name:** IncomeIQ

    **Machine Learning Algorithm:** Random Forest Classifier

    **Dataset:** Adult Census Income Dataset

    **Dataset Source:** Kaggle

    **Programming Language:** Python

    **Framework:** Streamlit

    **Data Processing:** Pandas

    **Data Visualization:** Plotly

    **Model Storage:** Joblib
    """)

    st.divider()

    # Model Information

    st.header("Model Information")

    st.write(
        "The Random Forest algorithm uses multiple decision trees "
        "to make predictions and combines their results to improve "
        "prediction performance."
    )

    st.metric(
        "Model Accuracy",
        "84.96%"
    )

    st.divider()

    st.caption(
        "IncomeIQ | Random Forest Machine Learning Project"
    )
