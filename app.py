import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateIQ | House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(255, 92, 40, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(255, 170, 70, 0.06),
                transparent 30%
            ),
            #090b10;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0c0f14;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.8rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-label {
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 99, 56, 0.10);
        border: 1px solid rgba(255, 99, 56, 0.25);
        color: #ff8a68;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 3.7rem;
        line-height: 1.02;
        font-weight: 850;
        letter-spacing: -2px;
        margin: 0;
    }

    .hero-title span {
        color: #ff6338;
    }

    .hero-subtitle {
        color: #9da5b2;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 900px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-kicker {
        color: #ff7048;
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .section-heading {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #9299a5;
        margin-bottom: 1.5rem;
    }


    /* ========================================================
       STREAMLIT CONTAINERS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.018)
            );
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.018)
            );
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8f97a4;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 13px !important;
        min-height: 48px;
        font-weight: 800 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(255,99,56,0.45) !important;
        box-shadow: 0 10px 30px rgba(255,99,56,0.12);
    }

    button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #ff5733,
                #ff7a45
            ) !important;
        color: white !important;
        border: none !important;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="input"] {
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }

    div[data-testid="stNumberInput"] input {
        font-weight: 650;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        font-weight: 750;
        font-size: 0.92rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff6b42 !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .hero-title {
            font-size: 2.6rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "model.joblib"
METADATA_PATH = "metadata.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(
            f"Could not load model.joblib: {str(e)}"
        )
        return None


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not os.path.exists(METADATA_PATH):
        return None

    try:
        return joblib.load(METADATA_PATH)

    except Exception as e:
        st.error(
            f"Could not load metadata.joblib: {str(e)}"
        )
        return None


# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()
metadata = load_metadata()


# ============================================================
# CHECK RESOURCES
# ============================================================

if model is None:

    st.error(
        "❌ model.joblib was not found. "
        "Please place model.joblib in the same folder as app.py."
    )

    st.stop()


if metadata is None:

    st.error(
        "❌ metadata.joblib was not found. "
        "Please place metadata.joblib in the same folder as app.py."
    )

    st.stop()


# ============================================================
# EXTRACT METADATA
# ============================================================

numeric_features = metadata.get(
    "numeric_features",
    []
)

categorical_features = metadata.get(
    "categorical_features",
    []
)

feature_names = metadata.get(
    "feature_names",
    []
)

numeric_defaults = metadata.get(
    "numeric_defaults",
    {}
)

categorical_defaults = metadata.get(
    "categorical_defaults",
    {}
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏠 EstateIQ")

    st.caption(
        "Intelligent Property Valuation"
    )

    st.divider()

    st.markdown("### 🟢 System Status")

    st.success(
        "Model loaded successfully"
    )

    st.divider()

    st.markdown("### 🧠 Model Stack")

    st.markdown(
        "**Algorithm**  \n"
        "Random Forest Regressor"
    )

    st.markdown(
        "**Optimization**  \n"
        "RandomizedSearchCV"
    )

    st.markdown(
        "**Validation**  \n"
        "5-Fold K-Fold Cross-Validation"
    )

    st.markdown(
        "**Preprocessing**  \n"
        "Pipeline + ColumnTransformer"
    )

    st.markdown(
        "**Encoding**  \n"
        "One-Hot Encoding"
    )

    st.markdown(
        "**Imputation**  \n"
        "Median + Most Frequent"
    )

    st.divider()

    st.markdown("### 📊 Model Details")

    st.caption(
        f"Model features: {len(feature_names)}"
    )

    st.caption(
        "Deployment: Streamlit"
    )

    st.caption(
        "Inference: Joblib"
    )

    st.divider()

    st.caption(
        "EstateIQ • Advanced House Price Regression"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-label">AI POWERED PROPERTY VALUATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-title">
        Advanced <span>House Price</span><br>
        Prediction
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        Turn property characteristics into an intelligent price estimate
        using an optimized Random Forest Regression pipeline.
        Built with preprocessing, validation, hyperparameter optimization,
        and production-ready deployment.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP INFO CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    with st.container(border=True):

        st.markdown("### 🌲")

        st.caption("ALGORITHM")

        st.markdown("**Random Forest**")


with c2:

    with st.container(border=True):

        st.markdown("### 🎯")

        st.caption("OPTIMIZATION")

        st.markdown("**Randomized Search**")


with c3:

    with st.container(border=True):

        st.markdown("### 🛡️")

        st.caption("VALIDATION")

        st.markdown("**5-Fold CV**")


with c4:

    with st.container(border=True):

        st.markdown("### ⚡")

        st.caption("DEPLOYMENT")

        st.markdown("**Streamlit**")


st.write("")
st.divider()
st.write("")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🏠  Single Prediction",
        "📂  Batch Prediction",
        "🧠  About EstateIQ"
    ]
)


# ============================================================
# TAB 1 — SINGLE PREDICTION
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-kicker">PROPERTY VALUATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">🏡 Predict House Price</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Enter the main characteristics of the property.
            Additional model features are automatically completed using
            values learned from the training data.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        with st.container(border=True):

            st.markdown("### 🏆 Property Quality")

            overall_qual = st.slider(
                "Overall Quality",
                min_value=1,
                max_value=10,
                value=7,
                help="Overall material and finish quality."
            )

            st.markdown("### 📐 Property Size")

            gr_liv_area = st.number_input(
                "Living Area (sq ft)",
                min_value=300,
                max_value=6000,
                value=1800,
                step=50
            )

            lot_area = st.number_input(
                "Lot Area (sq ft)",
                min_value=500,
                max_value=100000,
                value=8000,
                step=250
            )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.markdown("### 🏠 Structure")

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=2005,
                step=1
            )

            total_bsmt_sf = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0,
                max_value=5000,
                value=1000,
                step=50
            )

            first_floor_sf = st.number_input(
                "1st Floor Area (sq ft)",
                min_value=300,
                max_value=5000,
                value=1200,
                step=50
            )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        with st.container(border=True):

            st.markdown("### 🚗 Garage & Rooms")

            garage_cars = st.number_input(
                "Garage Capacity",
                min_value=0,
                max_value=5,
                value=2,
                step=1
            )

            garage_area = st.number_input(
                "Garage Area (sq ft)",
                min_value=0,
                max_value=1500,
                value=500,
                step=25
            )

            full_bath = st.number_input(
                "Full Bathrooms",
                min_value=0,
                max_value=5,
                value=2,
                step=1
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=3,
                step=1
            )

            total_rooms = st.number_input(
                "Total Rooms",
                min_value=1,
                max_value=20,
                value=7,
                step=1
            )

            fireplaces = st.number_input(
                "Fireplaces",
                min_value=0,
                max_value=5,
                value=1,
                step=1
            )


    st.write("")


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔮  Estimate Property Value",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Create input dictionary
            # ------------------------------------------------

            input_data = {}


            # ------------------------------------------------
            # Numerical defaults
            # ------------------------------------------------

            for feature in numeric_features:

                input_data[feature] = numeric_defaults.get(
                    feature,
                    np.nan
                )


            # ------------------------------------------------
            # Categorical defaults
            # ------------------------------------------------

            for feature in categorical_features:

                input_data[feature] = categorical_defaults.get(
                    feature,
                    "Missing"
                )


            # ------------------------------------------------
            # User values
            # ------------------------------------------------

            user_values = {

                "OverallQual": overall_qual,

                "GrLivArea": gr_liv_area,

                "YearBuilt": year_built,

                "LotArea": lot_area,

                "TotalBsmtSF": total_bsmt_sf,

                "1stFlrSF": first_floor_sf,

                "GarageCars": garage_cars,

                "GarageArea": garage_area,

                "FullBath": full_bath,

                "BedroomAbvGr": bedrooms,

                "TotRmsAbvGrd": total_rooms,

                "Fireplaces": fireplaces

            }


            # ------------------------------------------------
            # Override defaults
            # ------------------------------------------------

            for feature, value in user_values.items():

                if feature in input_data:

                    input_data[feature] = value


            # ------------------------------------------------
            # DataFrame
            # ------------------------------------------------

            input_df = pd.DataFrame(
                [input_data]
            )


            # ------------------------------------------------
            # Correct feature order
            # ------------------------------------------------

            if feature_names:

                input_df = input_df[
                    feature_names
                ]


            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = model.predict(
                input_df
            )[0]


            # =================================================
            # RESULT
            # =================================================

            st.write("")
            st.divider()
            st.write("")

            st.markdown(
                '<div class="section-kicker">AI VALUATION RESULT</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">💰 Estimated Property Value</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # BIG NATIVE METRIC
            # ------------------------------------------------

            result_col1, result_col2, result_col3 = st.columns(
                [1, 2, 1]
            )


            with result_col2:

                st.metric(
                    label="Estimated Sale Price",
                    value=f"${prediction:,.0f}"
                )


            st.success(
                "Prediction generated successfully using the trained "
                "Random Forest ML pipeline."
            )


            # =================================================
            # PREDICTION SUMMARY
            # =================================================

            st.write("")

            st.markdown(
                '<div class="section-kicker">PROPERTY PROFILE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">📋 Prediction Summary</div>',
                unsafe_allow_html=True
            )


            s1, s2, s3, s4 = st.columns(4)


            with s1:

                st.metric(
                    "🏆 Overall Quality",
                    overall_qual
                )


            with s2:

                st.metric(
                    "📐 Living Area",
                    f"{gr_liv_area:,} sq ft"
                )


            with s3:

                st.metric(
                    "🛏️ Bedrooms",
                    bedrooms
                )


            with s4:

                st.metric(
                    "🚿 Bathrooms",
                    full_bath
                )


            s5, s6, s7, s8 = st.columns(4)


            with s5:

                st.metric(
                    "📅 Year Built",
                    year_built
                )


            with s6:

                st.metric(
                    "🌳 Lot Area",
                    f"{lot_area:,} sq ft"
                )


            with s7:

                st.metric(
                    "🚗 Garage",
                    f"{garage_cars} cars"
                )


            with s8:

                st.metric(
                    "🔥 Fireplaces",
                    fireplaces
                )


        except Exception as e:

            st.error(
                f"❌ Prediction error: {str(e)}"
            )


# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-kicker">MULTI-PROPERTY ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">📂 Batch Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Upload a CSV containing property data and generate
            predictions for multiple houses at once.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "The uploaded CSV should contain the same model features "
        "used during training."
    )


    uploaded_file = st.file_uploader(
        "Upload Property Dataset",
        type=["csv"],
        help="Upload a CSV file containing house/property features."
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )


            # =================================================
            # DATASET OVERVIEW
            # =================================================

            st.markdown("### 📊 Dataset Overview")


            b1, b2, b3 = st.columns(3)


            with b1:

                st.metric(
                    "Properties",
                    f"{uploaded_df.shape[0]:,}"
                )


            with b2:

                st.metric(
                    "Columns",
                    f"{uploaded_df.shape[1]:,}"
                )


            with b3:

                st.metric(
                    "Model Features",
                    f"{len(feature_names):,}"
                )


            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )


            # =================================================
            # CHECK MISSING FEATURES
            # =================================================

            missing_features = [

                feature

                for feature in feature_names

                if feature not in uploaded_df.columns

            ]


            if missing_features:

                st.warning(
                    f"{len(missing_features)} model features are missing. "
                    "Learned defaults will be used for them."
                )

                with st.expander(
                    "🔎 Show missing features"
                ):

                    st.write(
                        missing_features
                    )


                # ---------------------------------------------
                # Fill numeric features
                # ---------------------------------------------

                for feature in numeric_features:

                    if feature not in uploaded_df.columns:

                        uploaded_df[feature] = numeric_defaults.get(
                            feature,
                            np.nan
                        )


                # ---------------------------------------------
                # Fill categorical features
                # ---------------------------------------------

                for feature in categorical_features:

                    if feature not in uploaded_df.columns:

                        uploaded_df[feature] = categorical_defaults.get(
                            feature,
                            "Missing"
                        )


            # =================================================
            # FEATURE ORDER
            # =================================================

            if feature_names:

                prediction_df = uploaded_df[
                    feature_names
                ]

            else:

                prediction_df = uploaded_df


            st.write("")


            # =================================================
            # GENERATE PREDICTIONS
            # =================================================

            generate_button = st.button(
                "🚀  Generate Property Valuations",
                type="primary",
                use_container_width=True
            )


            if generate_button:

                try:

                    predictions = model.predict(
                        prediction_df
                    )


                    result_df = uploaded_df.copy()


                    result_df[
                        "PredictedSalePrice"
                    ] = predictions


                    # -----------------------------------------
                    # Success
                    # -----------------------------------------

                    st.success(
                        f"Successfully generated "
                        f"{len(predictions):,} property predictions."
                    )


                    # -----------------------------------------
                    # Results
                    # -----------------------------------------

                    st.markdown(
                        "### 📈 Valuation Results"
                    )


                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


                    # -----------------------------------------
                    # Download
                    # -----------------------------------------

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")


                    st.download_button(
                        label="⬇️ Download Valuations CSV",
                        data=csv_data,
                        file_name="house_price_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


                except Exception as e:

                    st.error(
                        f"❌ Batch prediction error: {str(e)}"
                    )


        except Exception as e:

            st.error(
                f"❌ Error processing uploaded file: {str(e)}"
            )


# ============================================================
# TAB 3 — ABOUT ESTATEIQ
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-kicker">PROJECT ARCHITECTURE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">🧠 About EstateIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            An end-to-end Machine Learning application designed for
            intelligent property valuation.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROJECT INTRODUCTION
    # ========================================================

    with st.container(border=True):

        st.markdown("### 🏠 What is EstateIQ?")

        st.write(
            "EstateIQ is an end-to-end Machine Learning application "
            "for house price prediction. It combines data preprocessing, "
            "model validation, hyperparameter optimization and deployment "
            "into one complete workflow."
        )


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # MACHINE LEARNING PIPELINE
    # ========================================================

    st.markdown(
        '<div class="section-kicker">MACHINE LEARNING WORKFLOW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## 🔬 Machine Learning Pipeline"
    )

    st.write(
        "The project follows a structured workflow from raw data "
        "to deployed predictions."
    )


    # --------------------------------------------------------
    # PIPELINE ROW 1
    # --------------------------------------------------------

    p1, p2, p3, p4 = st.columns(4)


    with p1:

        with st.container(border=True):

            st.markdown("### 01")

            st.markdown("## 🔎")

            st.markdown("**Data Analysis**")

            st.caption(
                "Dataset structure, missing values, "
                "feature types and target distribution."
            )


    with p2:

        with st.container(border=True):

            st.markdown("### 02")

            st.markdown("## ⚙️")

            st.markdown("**Preprocessing**")

            st.caption(
                "Pipeline, ColumnTransformer, "
                "imputation and One-Hot Encoding."
            )


    with p3:

        with st.container(border=True):

            st.markdown("### 03")

            st.markdown("## 🌲")

            st.markdown("**Random Forest**")

            st.caption(
                "Nonlinear regression model designed "
                "to capture complex relationships."
            )


    with p4:

        with st.container(border=True):

            st.markdown("### 04")

            st.markdown("## 🏠")

            st.markdown("**Property Valuation**")

            st.caption(
                "Generate an estimated sale price "
                "from the trained model."
            )


    st.write("")


    # --------------------------------------------------------
    # PIPELINE ROW 2
    # --------------------------------------------------------

    p5, p6, p7, p8 = st.columns(4)


    with p5:

        with st.container(border=True):

            st.markdown("### 05")

            st.markdown("## 🧪")

            st.markdown("**Cross-Validation**")

            st.caption(
                "5-Fold K-Fold Cross-Validation "
                "for reliable model evaluation."
            )


    with p6:

        with st.container(border=True):

            st.markdown("### 06")

            st.markdown("## 🎯")

            st.markdown("**Optimization**")

            st.caption(
                "RandomizedSearchCV searches for "
                "strong Random Forest hyperparameters."
            )


    with p7:

        with st.container(border=True):

            st.markdown("### 07")

            st.markdown("## 📊")

            st.markdown("**Evaluation**")

            st.caption(
                "MAE, RMSE, R² and actual-vs-predicted "
                "analysis."
            )


    with p8:

        with st.container(border=True):

            st.markdown("### 08")

            st.markdown("## 🚀")

            st.markdown("**Deployment**")

            st.caption(
                "Final trained pipeline serialized "
                "with Joblib and deployed using Streamlit."
            )


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # MODEL ARCHITECTURE
    # ========================================================

    st.markdown(
        "## 🧠 Model Architecture"
    )


    model_col1, model_col2 = st.columns(2)


    with model_col1:

        with st.container(border=True):

            st.markdown("### 🔄 Preprocessing")

            st.markdown(
                "- Pipeline\n"
                "- ColumnTransformer\n"
                "- SimpleImputer\n"
                "- OneHotEncoder"
            )


    with model_col2:

        with st.container(border=True):

            st.markdown("### 🌲 Regression Model")

            st.markdown(
                "- Random Forest Regressor\n"
                "- Nonlinear relationships\n"
                "- Hyperparameter optimization\n"
                "- Production inference"
            )


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # HYPERPARAMETER OPTIMIZATION
    # ========================================================

    st.markdown(
        "## ⚙️ Hyperparameter Optimization"
    )

    st.write(
        "RandomizedSearchCV explored a search space containing:"
    )


    h1, h2, h3 = st.columns(3)


    with h1:

        with st.container(border=True):

            st.markdown("### 🌲 Tree Parameters")

            st.write("n_estimators")

            st.write("max_depth")


    with h2:

        with st.container(border=True):

            st.markdown("### 📏 Split Parameters")

            st.write("min_samples_split")

            st.write("min_samples_leaf")


    with h3:

        with st.container(border=True):

            st.markdown("### 🎛️ Model Parameters")

            st.write("max_features")

            st.write("bootstrap")


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # DATA LEAKAGE
    # ========================================================

    st.markdown(
        "## 🛡️ Data Leakage Prevention"
    )

    st.info(
        "Preprocessing is integrated directly into the Machine Learning "
        "pipeline. Transformations such as imputation and encoding are "
        "learned within the appropriate training folds during "
        "cross-validation."
    )


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # PROJECT STATS
    # ========================================================

    st.markdown(
        "## 📊 Project Overview"
    )


    stat1, stat2, stat3, stat4 = st.columns(4)


    with stat1:

        st.metric(
            "Training Samples",
            "1,460"
        )


    with stat2:

        st.metric(
            "Original Features",
            "80"
        )


    with stat3:

        st.metric(
            "CV Folds",
            "5"
        )


    with stat4:

        st.metric(
            "Final Model",
            "Random Forest"
        )


    st.write("")
    st.divider()
    st.write("")


    # ========================================================
    # FOOTER
    # ========================================================

    st.caption(
        "🏠 EstateIQ • Advanced House Price Regression • "
        "Machine Learning + Streamlit"
    )