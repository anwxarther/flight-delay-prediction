import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction System")

# =========================================
# LOAD MODELS
# =========================================

with open('weather_model.pkl', 'rb') as f:
    weather_model = pickle.load(f)

with open('delay_classifier.pkl', 'rb') as f:
    delay_classifier = pickle.load(f)

with open('delay_minutes_model.pkl', 'rb') as f:
    delay_minutes_model = pickle.load(f)

# =========================================
# LOAD COLUMNS
# =========================================

with open('weather_columns.pkl', 'rb') as f:
    weather_columns = pickle.load(f)

with open('delay_classifier_columns.pkl', 'rb') as f:
    delay_columns = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    minutes_columns = pickle.load(f)

# =========================================
# WEATHER SECTION
# =========================================

st.header("🌦 Weather Information")

use_api = st.toggle("Use Weather API")

# =========================================
# MANUAL WEATHER INPUT
# =========================================

if not use_api:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=15.00
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=60
    )

    wind_kph = st.number_input(
        "Wind Speed (kph)",
        min_value=0.0,
        max_value=200.0,
        value=15.0
    )

    gust_kph = st.number_input(
        "Wind Gust (kph)",
        min_value=0.0,
        max_value=250.0,
        value=20.0
    )

    visibility_km = st.number_input(
        "Visibility (km)",
        min_value=0.0,
        max_value=20.0,
        value=10.0
    )

    precip_mm = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0
    )
    
    pressure_mb = st.number_input(
        "Pressure (mb)",
        min_value=900.0,
        max_value=1100.0,
        value=1013.0
    )

# =========================================
# WEATHER API PLACEHOLDER
# =========================================

else:

    st.info("Weather API will be connected later.")

    temperature = 28.0
    humidity = 75
    wind_kph = 18.0
    gust_kph = 25.0
    visibility_km = 6.0
    precip_mm = 4.0
    pressure_mb = 1005.0

# =========================================
# WEATHER PREDICT BUTTON
# =========================================

weather_result = None

if st.button("🌤 Predict Weather Risk"):

    weather_input = pd.DataFrame([{
        'temperature_celsius': temperature,
        'humidity': humidity,
        'wind_kph': wind_kph,
        'gust_kph': gust_kph,
        'visibility_km': visibility_km,
        'precip_mm': precip_mm,
        'pressure_mb': pressure_mb
    }])

    weather_input = weather_input.reindex(
        columns=weather_columns,
        fill_value=0
    )

    weather_pred = weather_model.predict(weather_input)[0]

    weather_labels = {
        0: "Extreme",
        1: "Moderate",
        2: "Normal"
    }

    weather_result = weather_labels.get(weather_pred)

    st.session_state["weather_result"] = weather_result

    # =====================================
    # WEATHER DISPLAY
    # =====================================

    if weather_result == "Extreme":
        st.error(f"⛈ Weather Status: {weather_result}")

    elif weather_result == "Moderate":
        st.warning(f"🌥 Weather Status: {weather_result}")

    else:
        st.success(f"☀️ Weather Status: {weather_result}")

# =========================================
# KEEP WEATHER STATE
# =========================================

if "weather_result" in st.session_state:
    weather_result = st.session_state["weather_result"]

# =========================================
# FLIGHT SECTION
# =========================================

st.header("🛫 Flight Information")

dep_delay = st.number_input(
    "Departure Delay (minutes)",
    min_value=0,
    max_value=1000,
    value=0
)

distance = st.number_input(
    "Distance",
    min_value=0,
    max_value=20000,
    value=500
)

month = st.selectbox(
    "Month",
    list(range(1, 13))
)

day_of_week = st.selectbox(
    "Day Of Week",
    list(range(1, 8))
)

dep_time = st.number_input(
    "Departure Time (HHMM)",
    min_value=0,
    max_value=2359,
    value=900
)

# =========================================
# FLIGHT PREDICT BUTTON
# =========================================

if st.button("✈️ Predict Flight Delay"):

    # =====================================
    # WEATHER CHECK
    # =====================================

    if weather_result is None:

        st.warning("⚠️ Please predict weather first.")

    else:

        # =================================
        # FEATURE ENGINEERING
        # =================================

        log_distance = np.log1p(distance)

        is_weekend = 1 if day_of_week in [6, 7] else 0

        # =================================
        # CLASSIFIER INPUT
        # =================================

        delay_input = pd.DataFrame([{
            'DepDelay': dep_delay,
            'Distance': distance,
            'log_Distance': log_distance,
            'Month': month,
            'DayOfWeek': day_of_week,
            'DepTime': dep_time,
            'IsWeekend': is_weekend,
        }])

        delay_input = delay_input.reindex(
            columns=delay_columns,
            fill_value=0
        )

        # =================================
        # DELAY PROBABILITY
        # =================================

        delay_prob = delay_classifier.predict_proba(delay_input)[0][1]

        # =================================
        # WEATHER IMPACT
        # =================================

        if weather_result == "Moderate":
            delay_prob += 0.05

        elif weather_result == "Extreme":
            delay_prob += 0.10

        # =================================
        # FINAL DECISION
        # =================================

        threshold = 0.35

        delay_pred = 1 if delay_prob > threshold else 0

        # =================================
        # SHOW PROBABILITY
        # =================================

        st.write(
            f"### Delay Probability: {delay_prob * 100:.2f}%"
        )

        # =================================
        # DELAY OUTPUT
        # =================================

        if delay_pred == 1:

            st.error("⚠️ Flight Delay Expected")

            # =============================
            # MINUTES MODEL INPUT
            # =============================

            minutes_input = pd.DataFrame([{
                'DepDelay': dep_delay,
                'Distance': distance,
                'Month': month,
                'DayOfWeek': day_of_week,
                'DepTime': dep_time
            }])

            minutes_input = minutes_input.reindex(
                columns=minutes_columns,
                fill_value=0
            )

            base_delay = delay_minutes_model.predict(
                minutes_input
            )[0]

            # =============================
            # WEATHER ADJUSTMENT
            # =============================
            # classifier confidence impact
            final_delay = base_delay * (1 + delay_prob * 0.5)

            # weather impact
            if weather_result == "Moderate":
                final_delay *= 1.1

            elif weather_result == "Extreme":
                final_delay *= 1.3

            # long haul impact
            if distance >= 1500:
                final_delay *= 1.1

            # =============================
            # ROUNDING
            # =============================

            final_delay = round(final_delay)

            # =============================
            # FINAL DISPLAY
            # =============================

            st.subheader(
                f"⏰ Estimated Delay: {final_delay} minutes"
            )

        else:

            st.success("✅ Flight Likely On Time")