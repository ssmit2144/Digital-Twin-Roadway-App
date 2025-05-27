
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load data
gdf = pd.read_csv("pci_data_200_with_coords.csv")
model = joblib.load("pci_forecast_model.pkl")

# Sidebar filters
st.sidebar.title("Filters")
min_age = st.sidebar.slider("Min Age", int(gdf['age_years'].min()), int(gdf['age_years'].max()), int(gdf['age_years'].min()))
max_age = st.sidebar.slider("Max Age", min_age, int(gdf['age_years'].max()), int(gdf['age_years'].max()))

filtered = gdf[(gdf['age_years'] >= min_age) & (gdf['age_years'] <= max_age)]

# Predict future PCI
X = filtered[['ESALs', 'age_years']]
filtered['predicted_PCI'] = model.predict(X)

# Display map
st.title("Roadway PCI Digital Twin Viewer")
m = folium.Map(location=[filtered['latitude'].mean(), filtered['longitude'].mean()], zoom_start=12)

for _, row in filtered.iterrows():
    color = 'green' if row['predicted_PCI'] >= 70 else 'orange' if row['predicted_PCI'] >= 50 else 'red'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.7,
        tooltip=f"Segment: {row['segment_id']} | Predicted PCI: {row['predicted_PCI']:.1f}"
    ).add_to(m)

folium_static(m)

# Dataframe view
st.subheader("Segment Data")
st.dataframe(filtered[['segment_id', 'ESALs', 'age_years', 'current_PCI', 'predicted_PCI']])
