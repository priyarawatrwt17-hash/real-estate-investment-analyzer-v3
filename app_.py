import sys
import traceback
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Analyzer",
    page_icon="🏡",
    layout="wide"
)
st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #080D18;
    color: #E5EAF3;
}

/* Main headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #111C33;
    border: 1px solid #168BFF;
    padding: 20px;
    border-radius: 12px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #42A5FF !important;
}

/* Chart and content panels */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #263B5C;
    border-radius: 12px;
}

/* Input fields */
.stSelectbox, .stNumberInput, .stTextInput {
    color: #FFFFFF;
}

/* Buttons */
.stButton > button {
    background-color: #0878E8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #005CC8;
    color: white;
    border: 1px solid #42A5FF;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] a {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
/* ===== PREMIUM DASHBOARD ENHANCEMENTS ===== */

.stApp {
    background: linear-gradient(135deg, #080D18, #101B30);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101A2D;
    border-right: 1px solid #263B5C;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172943, #111C33);
    border: 1px solid #2876B8;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0, 120, 255, 0.12);
    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #42A5FF;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 120, 255, 0.22);
}

/* Section headings */
h1, h2, h3 {
    letter-spacing: 0.3px;
}

/* Chart panels */
[data-testid="stPlotlyChart"] {
    background: #111C33;
    border: 1px solid #263B5C;
    border-radius: 14px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Dividers */
hr {
    border-color: #263B5C;
}

[data-testid="stHeader"] {
    background-color: #0B1F3A !important;
    position: relative;
    height: 75px;
}

[data-testid="stHeader"]::after {
    content: "PROPERTY MARKET INTELLIGENCE\A DATA-DRIVEN INVESTMENT ANALYSIS";
    white-space: pre;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.8;
    letter-spacing: 1px;
    text-align: center;
}</style>
""", unsafe_allow_html=True)

# SIDEBAR BRANDING
st.sidebar.title("🏡 REAL ESTATE")
st.sidebar.caption("INVESTMENT ANALYZER")
st.sidebar.markdown("---")
st.sidebar.caption("Real Estate Market Intelligence")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "real_estate_model.pkl"
DATA_PATH = BASE_DIR / "cleaned_real_estate-.csv"

st.title("Real Estate Investment Analyzer")
st.warning(
    "The model predicts the dataset's rule-based category. "
    "It does not forecast actual property returns, rental yield, appreciation, or ROI."
)

# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep real_estate_model.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            "Keep cleaned_real_estate.csv in the same folder as app.py."
        )
    data = pd.read_csv(DATA_PATH)

    required = [
        "City", "Locality", "Property_Type", "BHK",
        "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt"
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data


try:
    model = load_model()
    df = load_data()
except Exception as exc:
    st.error(f"Unable to start the app: {type(exc).__name__}: {exc}")
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error(
        "The saved model does not expose feature_names_in_. "
        "The prediction input mapping must be checked against the training code."
    )
    st.stop()

features = list(model.feature_names_in_)
st.caption(f"Model loaded successfully · {len(features)} required model features")

# --------------------------------------------------
# 3. DATA QUALITY / SAFE NUMERIC PREPARATION
# --------------------------------------------------
numeric_columns = ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid_price = df["Price_in_Lakhs"].dropna()
valid_sqft_price = df["Price_per_SqFt"].dropna()
valid_size = df["Size_in_SqFt"].dropna()

# In this dataset, Price_per_SqFt is stored in LAKHS per sq.ft.
# Convert lakh/sq.ft. to rupees/sq.ft. for display.
avg_price_sqft_rupees = (
    valid_sqft_price.mean() * 100000 if not valid_sqft_price.empty else None
)

# INTERACTIVE DASHBOARD FILTERS
st.subheader("Market Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    city_options = ["All"] + sorted(df["City"].dropna().unique().tolist())
    selected_city = st.selectbox("Select City", city_options)

with filter_col2:
    property_options = ["All"] + sorted(df["Property_Type"].dropna().unique().tolist())
    selected_property = st.selectbox("Select Property Type", property_options)

with filter_col3:
    bhk_options = ["All"] + sorted(df["BHK"].dropna().unique().tolist())
    selected_bhk = st.selectbox("Select BHK", bhk_options)

# Apply selected filters
if selected_city != "All":
    df = df[df["City"] == selected_city]

if selected_property != "All":
    df = df[df["Property_Type"] == selected_property]

if selected_bhk != "All":
    df = df[df["BHK"] == selected_bhk]

# Recalculate KPI values using filtered data
valid_price = df["Price_in_Lakhs"].dropna()
valid_sqft_price = df["Price_per_SqFt"].dropna()
valid_size = df["Size_in_SqFt"].dropna()

avg_price_sqft_rupees = (
    valid_sqft_price.mean() * 100000
    if not valid_sqft_price.empty else None
)

# --------------------------------------------------
# 4. MARKET ANALYTICS DASHBOARD
# --------------------------------------------------
st.divider()
st.header("Real Estate Market Analytics Dashboard")
st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Properties", f"{len(df):,}")
k2.metric(
    "Average Property Price",
    f"₹{valid_price.mean():,.2f} Lakhs" if not valid_price.empty else "N/A"
)
k3.metric(
    "Average Price / Sq.Ft.",
    f"₹{avg_price_sqft_rupees:,.0f}" if avg_price_sqft_rupees is not None else "N/A"
)
k4.metric(
    "Average Property Size",
    f"{valid_size.mean():,.0f} Sq.Ft." if not valid_size.empty else "N/A"
)

st.caption(
    "Unit note: Price_per_SqFt is stored in lakhs per sq.ft. in the supplied dataset; "
    "the KPI converts it to rupees per sq.ft. (1 lakh = ₹100,000)."
)

# --------------------------------------------------
# MARKET INSIGHT CARDS
# --------------------------------------------------
st.subheader("Market Insights")

insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)

# 1. City with the highest number of listed properties
city_counts = df["City"].dropna().value_counts()

if not city_counts.empty:
    top_city = city_counts.index[0]
    city_share = (city_counts.iloc[0] / len(df)) * 100
else:
    top_city = "N/A"
    city_share = 0

# 2. Property type with the highest average price
type_prices = df.groupby("Property_Type")["Price_in_Lakhs"].mean().dropna()

if not type_prices.empty:
    premium_type = type_prices.idxmax()
    premium_price = type_prices.max()
else:
    premium_type = "N/A"
    premium_price = 0

# 3. Most common BHK configuration
bhk_counts = df["BHK"].dropna().value_counts()

if not bhk_counts.empty:
    popular_bhk = bhk_counts.index[0]
    bhk_share = (bhk_counts.iloc[0] / len(df)) * 100
else:
    popular_bhk = "N/A"
    bhk_share = 0

# 4. Highest-priced locality by average price
locality_prices = df.groupby("Locality")["Price_in_Lakhs"].mean().dropna()

if not locality_prices.empty:
    top_locality = locality_prices.idxmax()
    locality_price = locality_prices.max()
else:
    top_locality = "N/A"
    locality_price = 0

with insight_col1:
    st.metric("Leading City", str(top_city))
    st.caption(f"{city_share:.1f}% of filtered properties")

with insight_col2:
    st.metric("Premium Property Type", str(premium_type))
    st.caption(f"Average price: ₹{premium_price:,.2f} Lakhs")

with insight_col3:
    st.metric("Most Common Configuration", f"{popular_bhk} BHK" if popular_bhk != "N/A" else "N/A")
    st.caption(f"{bhk_share:.1f}% of filtered properties")

with insight_col4:
    st.metric("Top-Priced Locality", str(top_locality))
    st.caption(f"Average price: ₹{locality_price:,.2f} Lakhs")

st.divider()

st.divider()

# Two-column dashboard layout
left, right = st.columns(2)

# 1. Property Distribution by City
with left:
    st.subheader("Property Distribution by City")

    city_counts = (
        df["City"].fillna("Unknown")
        .value_counts()
        .head(10)
        .rename_axis("City")
        .reset_index(name="Number of Properties")
    )

    fig_city = px.bar(
        city_counts,
        x="City",
        y="Number of Properties",
        color="Number of Properties",
        color_continuous_scale=["#2563EB", "#38BDF8"],
        template="plotly_dark"
    )

    fig_city.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_city, use_container_width=True)


# 2. Average Price by Property Type
with right:
    st.subheader("Average Price by Property Type")

    type_price = (
        df.groupby("Property_Type", dropna=False)["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .rename_axis("Property Type")
        .reset_index(name="Average Price (₹ Lakhs)")
    )

    fig_type = px.bar(
        type_price,
        x="Property Type",
        y="Average Price (₹ Lakhs)",
        color="Average Price (₹ Lakhs)",
        color_continuous_scale=["#2563EB", "#38BDF8"],
        template="plotly_dark"
    )

    fig_type.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_type, use_container_width=True)


# Second row
left, right = st.columns(2)

# 3. Property Distribution by BHK
with left:
    st.subheader("Property Distribution by BHK")

    bhk_counts = (
        df["BHK"].dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("BHK")
        .reset_index(name="Number of Properties")
    )

    bhk_counts["BHK"] = bhk_counts["BHK"].astype(str) + " BHK"

    fig_bhk = px.bar(
        bhk_counts,
        x="BHK",
        y="Number of Properties",
        color="Number of Properties",
        color_continuous_scale=["#2563EB", "#38BDF8"],
        template="plotly_dark"
    )

    fig_bhk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_bhk, use_container_width=True)


# 4. Top 10 Localities by Average Property Price
with right:
    st.subheader("Top 10 Localities by Average Property Price")

    locality_price = (
        df.groupby("Locality", dropna=False)["Price_in_Lakhs"]
        .mean()
        .dropna()
        .nlargest(10)
        .sort_values(ascending=False)
        .rename_axis("Locality")
        .reset_index(name="Average Price (₹ Lakhs)")
    )

    fig_locality = px.bar(
        locality_price,
        x="Locality",
        y="Average Price (₹ Lakhs)",
        color="Average Price (₹ Lakhs)",
        color_continuous_scale=["#2563EB", "#38BDF8"],
        template="plotly_dark"
    )

    fig_locality.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_locality, use_container_width=True)

    st.caption(
        "The source dataset uses generated locality identifiers, "
        "not actual locality names."
    )
    
# --------------------------------------------------
# KEY BUSINESS FINDINGS
# --------------------------------------------------
st.divider()
st.header("Key Business Findings")
st.caption("Data-driven observations based on the current market filters")

# Calculate findings from the filtered dataset
if not df.empty:

    total_filtered = len(df)

    # City concentration
    city_summary = df["City"].value_counts()
    if not city_summary.empty:
        leading_city = city_summary.index[0]
        leading_city_share = city_summary.iloc[0] / total_filtered * 100
    else:
        leading_city = "N/A"
        leading_city_share = 0

    # Property type pricing gap
    type_summary = df.groupby("Property_Type")["Price_in_Lakhs"].mean().dropna()
    if not type_summary.empty:
        highest_type = type_summary.idxmax()
        highest_type_price = type_summary.max()
        lowest_type = type_summary.idxmin()
        lowest_type_price = type_summary.min()
        price_gap = highest_type_price - lowest_type_price
    else:
        highest_type = lowest_type = "N/A"
        highest_type_price = lowest_type_price = price_gap = 0

    # BHK distribution
    bhk_summary = df["BHK"].value_counts()
    if not bhk_summary.empty:
        leading_bhk = bhk_summary.index[0]
        leading_bhk_share = bhk_summary.iloc[0] / total_filtered * 100
    else:
        leading_bhk = "N/A"
        leading_bhk_share = 0

    # Locality pricing
    locality_summary = df.groupby("Locality")["Price_in_Lakhs"].mean().dropna()
    if not locality_summary.empty:
        highest_locality = locality_summary.idxmax()
        highest_locality_price = locality_summary.max()
        overall_avg_price = df["Price_in_Lakhs"].mean()
    else:
        highest_locality = "N/A"
        highest_locality_price = 0
        overall_avg_price = 0

    # Display findings
    finding_col1, finding_col2 = st.columns(2)

    with finding_col1:
        st.info(
            f"""
            **1. Geographic concentration**

            {leading_city} has the highest number of listed properties,
            representing {leading_city_share:.1f}% of the filtered dataset.

            **Business interpretation:** Property listings are most
            concentrated in this city within the selected data.
            """
        )

        st.info(
            f"""
            **2. Property pricing variation**

            {highest_type} has the highest average price at
            ₹{highest_type_price:,.2f} lakhs, while {lowest_type}
            has the lowest at ₹{lowest_type_price:,.2f} lakhs.

            **Price gap:** ₹{price_gap:,.2f} lakhs.

            **Business interpretation:** Average prices differ across
            property categories.
            """
        )

    with finding_col2:
        st.info(
            f"""
            **3. Popular property configuration**

            {leading_bhk} BHK is the most frequently listed configuration,
            accounting for {leading_bhk_share:.1f}% of filtered properties.

            **Business interpretation:** This is the most common
            configuration in the available listings.
            """
        )

        st.info(
            f"""
            **4. Locality price variation**

            {highest_locality} has the highest average listed price
            at ₹{highest_locality_price:,.2f} lakhs.

            Overall filtered average: ₹{overall_avg_price:,.2f} lakhs.

            **Business interpretation:** This locality has the highest
            average asking price in the selected dataset.
            """
        )

else:
    st.warning("No properties match the selected filters.")

# --------------------------------------------------
# 5. PROPERTY INPUT FORM
# --------------------------------------------------
st.divider()
st.header("Property Investment Classification")
st.caption(
    "Enter property details below. The model classifies the property using "
    "the patterns learned from the supplied dataset."
)

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
        bhk = st.number_input("BHK", min_value=1, max_value=20, value=2, step=1)
        size = st.number_input("Size (Sq.Ft.)", min_value=100, value=1200, step=50)
        current_year = pd.Timestamp.now().year
        year_built = st.number_input(
            "Year Built", min_value=1900, max_value=current_year,
            value=min(2020, current_year), step=1
        )

    with col2:
        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-furnished", "Furnished"]
        )
        floor_no = st.number_input("Floor No.", min_value=0, value=2, step=1)
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
            "Public Transport Accessibility", ["High", "Medium", "Low"]
        )
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])

    with col3:
        facing = st.selectbox(
            "Facing", ["North", "South", "East", "West"]
        )
        # These choices match the category values in the supplied dataset.
        owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
        availability = st.selectbox(
            "Availability Status",
            ["Ready_to_Move", "Under_Construction"],
            format_func=lambda x: x.replace("_", " ")
        )
        amenities = st.multiselect(
            "Amenities",
            ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
        )

    submitted = st.form_submit_button("Analyze Investment", type="primary")

# --------------------------------------------------
# 6. PREPARE INPUT AND PREDICT
# --------------------------------------------------
if submitted:
    if floor_no > total_floors:
        st.error("Floor No. cannot be greater than Total Floors. Please correct it.")
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

    missing_features = [feature for feature in features if feature not in values]

    if missing_features:
        st.error(
            "The app cannot create all required model inputs: "
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
            st.subheader("Prediction Result")

            if prediction == 1:
                st.success(
                    "Good Investment — according to the dataset's label rule."
                )
            else:
                st.info(
                    "Not classified as Good Investment according to the dataset's label rule."
                )

            st.caption(
                "This is a classification against the dataset's rule-based label. "
                "It is not a verified forecast of rental yield, appreciation, profit, or ROI."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())

