
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Market Analysis",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #080D18;
    color: #E5EAF3;
}

h1, h2, h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

[data-testid="stSidebar"] * {
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
[data-testid="stHeader"] {
    background-color: #080D18 !important;
}

[data-testid="stToolbar"] {
    background-color: #080D18 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "cleaned_real_estate-.csv"
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "City",
        "Property_Type",
        "BHK",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Size_in_SqFt"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    for col in [
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Size_in_SqFt",
        "BHK"
    ]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    return data.dropna(
        subset=[
            "City",
            "Property_Type",
            "BHK",
            "Price_in_Lakhs",
            "Price_per_SqFt",
            "Size_in_SqFt"
        ]
    )

try:
    df = load_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

# --------------------------------------------------
# 4. PAGE HEADER
# --------------------------------------------------
st.markdown("""
<div style="
    background-color:#071B49;
    padding:18px 30px;
    text-align:right;
    margin-bottom:25px;
    border-radius:8px;">
    <h4 style="color:white; margin:0;">
        PROPERTY MARKET INTELLIGENCE
    </h4>
    <h4 style="color:white; margin:5px 0 0 0;">
        DATA-DRIVEN INVESTMENT ANALYSIS
    </h4>
</div>
""", unsafe_allow_html=True)

st.title("📊 Market Analysis")
st.caption(
    "Analyze property pricing, market segments "
    "and residential property trends."
)

st.divider()

# --------------------------------------------------
# 5. MARKET FILTERS
# --------------------------------------------------
st.subheader("🔎 Market Filters")

col1, col2, col3 = st.columns(3)

with col1:
    city_options = ["All"] + sorted(
        df["City"].dropna().unique().tolist()
    )
    selected_city = st.selectbox(
        "Select City", city_options
    )

with col2:
    type_options = ["All"] + sorted(
        df["Property_Type"].dropna().unique().tolist()
    )
    selected_type = st.selectbox(
        "Select Property Type", type_options
    )

with col3:
    bhk_options = ["All"] + sorted(
        df["BHK"].dropna().unique().tolist()
    )
    selected_bhk = st.selectbox(
        "Select BHK", bhk_options
    )

filtered = df.copy()

if selected_city != "All":
    filtered = filtered[
        filtered["City"] == selected_city
    ]

if selected_type != "All":
    filtered = filtered[
        filtered["Property_Type"] == selected_type
    ]

if selected_bhk != "All":
    filtered = filtered[
        filtered["BHK"] == selected_bhk
    ]

if filtered.empty:
    st.warning("No properties match the selected filters.")
    st.stop()

st.divider()

# --------------------------------------------------
# 6. MARKET PERFORMANCE KPIs
# --------------------------------------------------
st.subheader("📌 Market Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Properties Analyzed",
    f"{len(filtered):,}"
)

k2.metric(
    "Average Property Price",
    f"₹{filtered['Price_in_Lakhs'].mean():,.2f} L"
)

k3.metric(
    "Average Price / Sq.Ft.",
    f"₹{filtered['Price_per_SqFt'].mean() * 100000:,.0f}"
)

k4.metric(
    "Average Property Size",
    f"{filtered['Size_in_SqFt'].mean():,.0f} Sq.Ft."
)

st.caption(
    "Price per sq.ft. is converted from lakh rupees "
    "per sq.ft. to rupees per sq.ft."
)

st.divider()

# --------------------------------------------------
# 7. CITY-WISE PRICE ANALYSIS
# --------------------------------------------------
st.subheader("🏙️ City-wise Property Price Analysis")

city_price = (
    filtered.groupby("City", as_index=False)
    ["Price_in_Lakhs"].mean()
    .sort_values("Price_in_Lakhs", ascending=False)
    .head(15)
)

fig1 = px.bar(
    city_price,
    x="City",
    y="Price_in_Lakhs",
    title="Top 15 Cities by Average Property Price",
    labels={
        "City": "City",
        "Price_in_Lakhs": "Average Price (₹ Lakhs)"
    },
    color="Price_in_Lakhs",
    color_continuous_scale="Blues"
)

fig1.update_layout(
    template="plotly_dark",
    xaxis_tickangle=-35
)

st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --------------------------------------------------
# 8. PROPERTY TYPE ANALYSIS
# --------------------------------------------------
st.subheader("🏠 Property Type Price Comparison")

type_price = (
    filtered.groupby("Property_Type", as_index=False)
    ["Price_in_Lakhs"].mean()
    .sort_values("Price_in_Lakhs", ascending=False)
)

fig2 = px.bar(
    type_price,
    x="Property_Type",
    y="Price_in_Lakhs",
    title="Average Price by Property Type",
    labels={
        "Property_Type": "Property Type",
        "Price_in_Lakhs": "Average Price (₹ Lakhs)"
    },
    color="Price_in_Lakhs",
    color_continuous_scale="Teal"
)

fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --------------------------------------------------
# 9. BHK-WISE ANALYSIS
# --------------------------------------------------
st.subheader("🛏️ BHK-wise Property Price Analysis")

bhk_price = (
    filtered.groupby("BHK", as_index=False)
    ["Price_in_Lakhs"].mean()
    .sort_values("BHK")
)

fig3 = px.bar(
    bhk_price,
    x="BHK",
    y="Price_in_Lakhs",
    title="Average Property Price by BHK",
    labels={
        "BHK": "BHK Configuration",
        "Price_in_Lakhs": "Average Price (₹ Lakhs)"
    },
    color="Price_in_Lakhs",
    color_continuous_scale="Purples"
)

fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --------------------------------------------------
# 10. PRICE DISTRIBUTION
# --------------------------------------------------
st.subheader("💰 Property Price Distribution")

fig4 = px.histogram(
    filtered,
    x="Price_in_Lakhs",
    nbins=40,
    title="Distribution of Property Prices",
    labels={
        "Price_in_Lakhs": "Property Price (₹ Lakhs)"
    }
)

fig4.update_layout(template="plotly_dark")

st.plotly_chart(fig4, use_container_width=True)

st.divider()

# --------------------------------------------------
# 11. PRICE PER SQ.FT. ANALYSIS
# --------------------------------------------------
st.subheader("📐 Price per Square Foot Analysis")

fig5 = px.box(
    filtered,
    x="Property_Type",
    y="Price_per_SqFt",
    title="Price per Sq.Ft. Distribution by Property Type",
    labels={
        "Property_Type": "Property Type",
        "Price_per_SqFt": "Price per Sq.Ft. (₹ Lakhs)"
    }
)

fig5.update_layout(template="plotly_dark")

st.plotly_chart(fig5, use_container_width=True)

st.caption(
    "The box plot shows the distribution and variation "
    "of price per square foot across property types."
)

st.divider()

st.caption(
    "Analysis is based on the supplied property listing dataset. "
    "Observed prices are not verified transaction prices."
)
