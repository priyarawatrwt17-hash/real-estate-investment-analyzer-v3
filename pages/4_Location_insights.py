
import streamlit as st
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071B49 0%, #063B78 100%);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar navigation */
[data-testid="stSidebarNav"] {
    background: transparent;
}

/* Selected navigation item */
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: rgba(0, 180, 255, 0.25);
    border-radius: 8px;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25);
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
}
</style>
""", unsafe_allow_html=True)
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Location Insights",
    page_icon="📍",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07152E 0%, #102B5C 55%, #164A91 100%);
    color: #F4F8FF;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #F4F8FF;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #102B5C, #174B91);
    border: 1px solid #26B5FC;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #A9C9E8;
}

div[data-testid="stMetricValue"] {
    color: #35C5FF;
}

div[data-baseweb="select"] > div {
    background-color: #102B5C;
    color: white;
    border-color: #26B5FC;
}

hr {
    border-color: #263B5C;
}
</style>
""", unsafe_allow_html=True)


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "cleaned_real_estate-.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

    st.title("📍 Location Insights")
    st.caption("Explore property distribution and price patterns across cities and localities.")

    # City filter
    cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
    selected_city = st.selectbox("Select City", cities)

    if selected_city != "All":
        filtered_df = df[df["City"] == selected_city].copy()
    else:
        filtered_df = df.copy()

    st.divider()

    # Location KPIs
    total_listings = len(filtered_df)
    city_count = filtered_df["City"].nunique()
    avg_price = filtered_df["Price_in_Lakhs"].mean()
    avg_price_sqft = filtered_df["Price_per_SqFt"].mean() * 100000

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Property Listings", f"{total_listings:,}")
    c2.metric("Cities Covered", f"{city_count:,}")
    c3.metric("Average Property Price", f"₹{avg_price:,.2f} Lakhs")
    c4.metric("Average Price / Sq. Ft.", f"₹{avg_price_sqft:,.0f}")

    st.divider()

    # City analysis
    st.subheader("City-wise Market Distribution")

    city_data = (
        filtered_df.groupby("City")
        .size()
        .reset_index(name="Property Listings")
        .sort_values("Property Listings", ascending=False)
    )

    fig_city = px.bar(
        city_data,
        x="City",
        y="Property Listings",
        title="Number of Property Listings by City",
        text="Property Listings"
    )

    fig_city.update_layout(
        template="plotly_dark",
        xaxis_title="City",
        yaxis_title="Number of Listings"
    )

    st.plotly_chart(fig_city, use_container_width=True)

    st.divider()

    # Locality analysis
    st.subheader("Locality-wise Price Analysis")

    locality_data = (
        filtered_df.groupby("Locality")
        .agg(
            Average_Price=("Price_in_Lakhs", "mean"),
            Property_Listings=("Price_in_Lakhs", "count")
        )
        .reset_index()
    )

    locality_data = locality_data.sort_values(
        "Average_Price", ascending=False
    ).head(10)

    fig_locality = px.bar(
        locality_data,
        x="Locality",
        y="Average_Price",
        title="Top 10 Localities by Average Property Price",
        text_auto=".2f"
    )

    fig_locality.update_layout(
        template="plotly_dark",
        xaxis_title="Locality",
        yaxis_title="Average Price (₹ Lakhs)"
    )

    st.plotly_chart(fig_locality, use_container_width=True)

    st.divider()

    # Locality listing concentration
    st.subheader("Property Listing Concentration by Locality")

    listing_data = (
        filtered_df["Locality"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    listing_data.columns = ["Locality", "Property Listings"]

    fig_listings = px.bar(
        listing_data,
        x="Locality",
        y="Property Listings",
        title="Top 10 Localities by Number of Listings",
        text="Property Listings"
    )

    fig_listings.update_layout(
        template="plotly_dark",
        xaxis_title="Locality",
        yaxis_title="Number of Listings"
    )

    st.plotly_chart(fig_listings, use_container_width=True)

    st.caption(
        "Note: The supplied dataset uses generated locality identifiers, "
        "not verified real-world locality names."
    )

except Exception as e:
    st.error(f"Unable to load location data: {e}")
