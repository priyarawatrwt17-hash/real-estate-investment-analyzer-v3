
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Investment Insights",
    page_icon="📊",
    layout="wide"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #080D18;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #000000;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF;
}
h1, h2, h3, p, label {
    color: #FFFFFF !important;
}
div[data-testid="stMetric"] {
    background-color: #111C30;
    border: 1px solid #263B5C;
    padding: 18px;
    border-radius: 12px;
}

/* Match the top header with the dark navy theme */
[data-testid="stHeader"] {
    background-color: #080D18 !important;
}

[data-testid="stToolbar"] {
    background-color: #080D18 !important;
}

[data-testid="stDecoration"] {
    background-image: none !important;
    background-color: #080D18 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
DATA_PATH = Path(__file__).resolve().parent.parent / "cleaned_real_estate-.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.title("📊 Investment Insights")
st.caption("Data-driven observations from the real estate property dataset.")
st.divider()

# ---------------- DATA PREPARATION ----------------
required_columns = ["City", "Property_Type", "BHK", "Price_in_Lakhs",
                    "Size_in_SqFt"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Required columns missing from dataset: {missing}")
    st.stop()

data = df.copy()

for col in ["Price_in_Lakhs", "Size_in_SqFt", "BHK"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["City", "Property_Type",
                           "Price_in_Lakhs", "Size_in_SqFt"])

# ---------------- FILTERS ----------------
st.subheader("Explore the Market")

c1, c2 = st.columns(2)

cities = ["All"] + sorted(data["City"].dropna().astype(str).unique())

with c1:
    selected_city = st.selectbox("Select City", cities)

property_types = ["All"] + sorted(
    data["Property_Type"].dropna().astype(str).unique()
)

with c2:
    selected_type = st.selectbox("Property Type", property_types)

filtered = data.copy()

if selected_city != "All":
    filtered = filtered[filtered["City"].astype(str) == selected_city]

if selected_type != "All":
    filtered = filtered[
        filtered["Property_Type"].astype(str) == selected_type
    ]

if filtered.empty:
    st.warning("No properties match the selected filters.")
    st.stop()

st.divider()

# ---------------- KPI CARDS ----------------
st.subheader("Market Snapshot")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Properties Analysed", f"{len(filtered):,}")
k2.metric("Average Price", f"₹{filtered['Price_in_Lakhs'].mean():,.2f} L")
k3.metric("Average Size", f"{filtered['Size_in_SqFt'].mean():,.0f} sq ft")
k4.metric("Average Price / Sq Ft",
          f"₹{(filtered['Price_in_Lakhs'].mean() * 100000 / filtered['Size_in_SqFt'].mean()):,.0f}")

st.divider()

# ---------------- INSIGHT 1 ----------------
st.subheader("1. City-wise Price Comparison")

city_prices = (
    filtered.groupby("City", as_index=False)
    .agg(Average_Price=("Price_in_Lakhs", "mean"))
    .sort_values("Average_Price", ascending=False)
)

fig1 = px.bar(
    city_prices,
    x="City",
    y="Average_Price",
    color="Average_Price",
    title="Average Property Price by City",
    template="plotly_dark"
)
st.plotly_chart(fig1, use_container_width=True)

top_city = city_prices.iloc[0]
st.info(
    f"{top_city['City']} has the highest average listed property price "
    f"among the cities in the current selection "
    f"(₹{top_city['Average_Price']:.2f} lakhs)."
)

# ---------------- INSIGHT 2 ----------------
st.subheader("2. Property Type Analysis")

type_prices = (
    filtered.groupby("Property_Type", as_index=False)
    .agg(Average_Price=("Price_in_Lakhs", "mean"),
         Listings=("Price_in_Lakhs", "count"))
    .sort_values("Average_Price", ascending=False)
)

fig2 = px.bar(
    type_prices,
    x="Property_Type",
    y="Average_Price",
    color="Property_Type",
    title="Average Price by Property Type",
    template="plotly_dark"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- INSIGHT 3 ----------------
st.subheader("3. Property Configuration Analysis")

if filtered["BHK"].notna().any():
    bhk_data = (
        filtered.groupby("BHK", as_index=False)
        .size()
        .rename(columns={"size": "Listings"})
        .sort_values("BHK")
    )

    fig3 = px.bar(
        bhk_data,
        x="BHK",
        y="Listings",
        title="Property Listings by BHK Configuration",
        template="plotly_dark"
    )
    st.plotly_chart(fig3, use_container_width=True)

    popular_bhk = bhk_data.loc[bhk_data["Listings"].idxmax()]
    st.info(
        f"{int(popular_bhk['BHK'])} BHK is the most frequently listed "
        f"configuration in the current selection."
    )

# ---------------- INSIGHT 4 ----------------
st.subheader("4. Price vs Property Size")

fig4 = px.scatter(
    filtered,
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    color="Property_Type",
    hover_data=["City", "BHK"],
    title="Relationship Between Property Size and Listed Price",
    template="plotly_dark"
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------------- BUSINESS INTERPRETATION ----------------
st.subheader("5. Business Interpretation")

st.markdown("""
- **Geographic comparison:** Compare average listed prices across cities
  to identify differences in market positioning.
- **Property segmentation:** Compare property types to understand
  differences in average asking prices.
- **Configuration demand proxy:** Listing counts by BHK show which
  configurations are more common in this dataset. They do not prove buyer demand.
- **Size and pricing:** The scatter plot helps examine how listed prices
  vary with property size.
""")

st.warning(
    "These insights describe the supplied property listing dataset. "
    "Listed prices are not verified transaction prices. "
    "They do not establish actual rental yield, appreciation, ROI or profit."
)

st.caption("Real Estate Investment Analyzer | Investment Insights")
