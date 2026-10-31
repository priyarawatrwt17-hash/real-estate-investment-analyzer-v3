
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Property Explorer",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "cleaned_real_estate-.csv"
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Convert numeric columns safely
    numeric_columns = [
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Size_in_SqFt",
        "BHK"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


try:
    df = load_data()

except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# --------------------------------------------------
# 3. PAGE STYLING
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        120deg,
        #071B49 0%,
        #063B78 55%,
        #0878C9 100%
    );
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

h1, h2, h3, p, label {
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background-color: #111C33;
    border: 1px solid #168BFF;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

[data-testid="stMetricLabel"] {
    color: #B8D9FF !important;
}

.stDownloadButton button {
    background-color: #0878C9;
    color: white;
    border-radius: 8px;
    border: none;
}
header[data-testid="stHeader"] {
    background-color: #071B49 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. PAGE TITLE
# --------------------------------------------------

st.title("🏡 Property Explorer")

st.markdown(
    "Explore and filter real estate properties "
    "based on location, configuration, price and size."
)

st.divider()


# --------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "City",
    "Property_Type",
    "BHK",
    "Price_in_Lakhs",
    "Size_in_SqFt"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required dataset columns: {missing_columns}"
    )
    st.stop()


# --------------------------------------------------
# 6. FILTER SECTION
# --------------------------------------------------

st.subheader("🔎 Property Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    city_options = ["All"] + sorted(
        df["City"].dropna().astype(str).unique().tolist()
    )

    selected_city = st.selectbox(
        "Select City",
        city_options
    )

with filter_col2:
    property_options = ["All"] + sorted(
        df["Property_Type"].dropna().astype(str).unique().tolist()
    )

    selected_property = st.selectbox(
        "Select Property Type",
        property_options
    )

with filter_col3:
    bhk_values = sorted(
        df["BHK"].dropna().unique().tolist()
    )

    bhk_options = ["All"] + bhk_values

    selected_bhk = st.selectbox(
        "Select BHK",
        bhk_options
    )


# --------------------------------------------------
# 7. PRICE AND SIZE FILTERS
# --------------------------------------------------

st.markdown("### Price and Property Size")

price_col, size_col = st.columns(2)

valid_prices = df["Price_in_Lakhs"].dropna()
valid_sizes = df["Size_in_SqFt"].dropna()

if valid_prices.empty or valid_sizes.empty:
    st.error("Price or property size data is unavailable.")
    st.stop()

with price_col:
    min_price = float(valid_prices.min())
    max_price = float(valid_prices.max())

    selected_price = st.slider(
        "Price Range (₹ Lakhs)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

with size_col:
    min_size = float(valid_sizes.min())
    max_size = float(valid_sizes.max())

    selected_size = st.slider(
        "Property Size (Sq.Ft.)",
        min_value=min_size,
        max_value=max_size,
        value=(min_size, max_size)
    )


# --------------------------------------------------
# 8. APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["City"].astype(str) == selected_city
    ]

if selected_property != "All":
    filtered_df = filtered_df[
        filtered_df["Property_Type"].astype(str)
        == selected_property
    ]

if selected_bhk != "All":
    filtered_df = filtered_df[
        filtered_df["BHK"] == selected_bhk
    ]

filtered_df = filtered_df[
    filtered_df["Price_in_Lakhs"].between(
        selected_price[0], selected_price[1]
    )
]

filtered_df = filtered_df[
    filtered_df["Size_in_SqFt"].between(
        selected_size[0], selected_size[1]
    )
]


# --------------------------------------------------
# 9. DYNAMIC KPI CARDS
# --------------------------------------------------

st.divider()
st.subheader("📊 Property Overview")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        "Matching Properties",
        f"{len(filtered_df):,}"
    )

with kpi2:
    avg_price = filtered_df["Price_in_Lakhs"].mean()

    st.metric(
        "Average Property Price",
        f"₹ {avg_price:,.2f} Lakhs"
        if pd.notna(avg_price) else "N/A"
    )

with kpi3:
    avg_size = filtered_df["Size_in_SqFt"].mean()

    st.metric(
        "Average Property Size",
        f"{avg_size:,.0f} Sq.Ft."
        if pd.notna(avg_size) else "N/A"
    )


# --------------------------------------------------
# 10. FILTERED PROPERTY RECORDS
# --------------------------------------------------

st.divider()
st.subheader("🏘️ Matching Property Listings")

st.write(
    f"Showing {min(len(filtered_df), 100)} "
    f"of {len(filtered_df):,} matching properties."
)

if not filtered_df.empty:

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    # Download all filtered records
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Properties",
        data=csv_data,
        file_name="filtered_real_estate_properties.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "No properties match the selected filters. "
        "Please adjust your filter selections."
    )


# --------------------------------------------------
# 11. END OF PAGE
# --------------------------------------------------

st.divider()

st.caption(
    "Real Estate Investment Analyzer | Property Explorer"
)
