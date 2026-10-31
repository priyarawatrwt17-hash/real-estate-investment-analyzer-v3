
import traceback
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.title("🏡 Investment Classifier")
st.caption("Evaluate a property using the trained classification model.")

st.markdown("""
<style>
header[data-testid="stHeader"] {
    background-color: #071B49 !important;
}
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #000000 !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
.stApp {
    background-color: #080D18;
    color: #E5EAF3;
}
h1, h2, h3 {
    color: #FFFFFF !important;
}
[data-testid="stMetric"] {
    background-color: #111C33;
    border: 1px solid #168BFF;
    padding: 20px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #42A5FF !important;
}
.stButton > button {
    background-color: #0878E8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "real_estate_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "real_estate_model.pkl must be in the main project folder."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("The saved model does not expose its required feature names.")
    st.stop()

features = list(model.feature_names_in_)
st.success(f"Model loaded successfully · {len(features)} required features")

st.warning(
    "This model predicts the dataset's rule-based category. "
    "It does not forecast actual property returns, rental yield, "
    "appreciation, or ROI."
)

# --------------------------------------------------
# PROPERTY INPUT FORM
# --------------------------------------------------
st.divider()
st.header("Enter Property Details")

with st.form("property_form"):
    st.subheader("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.text_input("State", value="Uttarakhand")
        city = st.text_input("City", value="Dehradun")
        locality = st.text_input("Locality", value="Central")

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Independent House", "Villa"]
        )

        bhk = st.number_input(
            "BHK", min_value=1, max_value=20, value=2, step=1
        )

        size = st.number_input(
            "Size (Sq.Ft.)", min_value=100, value=1200, step=50
        )

        current_year = pd.Timestamp.now().year

        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=current_year,
            value=min(2020, current_year),
            step=1
        )

    with col2:
        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-furnished", "Furnished"]
        )

        floor_no = st.number_input(
            "Floor No.", min_value=0, value=2, step=1
        )

        total_floors = st.number_input(
            "Total Floors", min_value=1, value=5, step=1
        )

        nearby_schools = st.number_input(
            "Nearby Schools", min_value=0, value=3, step=1
        )

        nearby_hospitals = st.number_input(
            "Nearby Hospitals", min_value=0, value=2, step=1
        )

        transport = st.selectbox(
            "Public Transport Accessibility",
            ["High", "Medium", "Low"]
        )

        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])

    with col3:
        facing = st.selectbox(
            "Facing", ["North", "South", "East", "West"]
        )

        owner_type = st.selectbox(
            "Owner Type", ["Owner", "Builder", "Broker"]
        )

        availability = st.selectbox(
            "Availability Status",
            ["Ready_to_Move", "Under_Construction"],
            format_func=lambda x: x.replace("_", " ")
        )

        amenities = st.multiselect(
            "Amenities",
            ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
        )

    submitted = st.form_submit_button(
        "Analyze Investment",
        type="primary"
    )

# --------------------------------------------------
# PREPARE INPUT AND PREDICT
# --------------------------------------------------
if submitted:

    if floor_no > total_floors:
        st.error(
            "Floor No. cannot be greater than Total Floors. "
            "Please correct it."
        )
        st.stop()

    floor_excess = floor_no - total_floors
    floor_issue = int(floor_excess > 0)

    school_score = min(nearby_schools, 10)
    hospital_score = min(nearby_hospitals, 10)

    values = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": current_year - year_built,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Floor_Issue_Flag": floor_issue,
        "Amenities_Standardized": ", ".join(amenities),
        "Floor_Quality_Status": "Inconsistent" if floor_issue else "Valid",
        "Floor_Excess": floor_excess,
        "Floor_Difference": floor_excess,
        "Floor_Relationship": (
            "Floor_No above Total_Floors"
            if floor_issue else "Floor_No within Total_Floors"
        ),
        "Floor_Above_Total": floor_issue,
        "School_Density_Score": school_score,
        "Hospital_Accessibility_Score": hospital_score,
        "Has_Gym": int("Gym" in amenities),
        "Has_Garden": int("Garden" in amenities),
        "Has_Pool": int("Pool" in amenities),
        "Has_Clubhouse": int("Clubhouse" in amenities),
        "Has_Playground": int("Playground" in amenities),
    }

    missing_features = [
        feature for feature in features if feature not in values
    ]

    if missing_features:
        st.error(
            "Missing required model inputs: "
            + ", ".join(missing_features)
        )

    else:
        input_row = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features
        )

        with st.expander("Review model input values"):
            st.dataframe(input_row, use_container_width=True)

        try:
            prediction = int(model.predict(input_row)[0])

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.success(
                    "Good Investment — according to the dataset's label rule."
                )
            else:
                st.info(
                    "Not classified as Good Investment according to "
                    "the dataset's label rule."
                )

            st.caption(
                "This is a classification against the dataset's "
                "rule-based label. It is not a verified forecast of "
                "rental yield, appreciation, profit, or ROI."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())
