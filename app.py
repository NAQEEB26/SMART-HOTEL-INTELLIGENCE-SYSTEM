"""
Smart Hotel Intelligence System
================================
A data science product that helps hotel managers:
  1. Predict booking cancellation risk before it happens
  2. Estimate optimal room revenue for any booking
  3. Explore data-driven analytics across hotel performance

Built with: Python · Streamlit · Scikit-learn · Plotly
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Hotel Intelligence System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .app-header h1 { color: white !important; font-size: 2.2rem; margin: 0; }
    .app-header p  { color: #aab4c8; font-size: 1.05rem; margin: 0.4rem 0 0 0; }

    .info-card {
        background: #f8f9ff;
        border: 1px solid #e0e4f0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .info-card h4 { color: #1a1a2e; margin-top: 0; margin-bottom: 0.5rem; }

    .result-high {
        background: linear-gradient(135deg, #ff6b6b22, #ff4d4d11);
        border: 2px solid #ff4d4d;
        border-radius: 12px; padding: 1.4rem; text-align: center;
    }
    .result-medium {
        background: linear-gradient(135deg, #ffa50222, #ff8c0011);
        border: 2px solid #ffa502;
        border-radius: 12px; padding: 1.4rem; text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #2ed57322, #00c85111);
        border: 2px solid #2ed573;
        border-radius: 12px; padding: 1.4rem; text-align: center;
    }

    div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}
MODEL_DIR = 'model'


def _find_data_path():
    for p in [
        'hotel_booking.csv',
        'data/hotel_booking.csv',
        '../hotel_booking.csv',
        'data/hotel_booking_sample.csv',
        '../data/hotel_booking_sample.csv',
    ]:
        if os.path.exists(p):
            return p
    return None


DATA_PATH = _find_data_path()

# ── Loaders (cached) ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    # Auto-train if model files are missing (e.g. first run on Streamlit Cloud)
    missing = not all(
        os.path.exists(f'{MODEL_DIR}/{f}')
        for f in ['cancel_model.pkl', 'adr_model.pkl', 'label_encoders.pkl', 'feature_info.pkl']
    )
    if missing:
        with st.spinner("⏳ First-time setup: Downloading dataset & training models (3–5 min)..."):
            try:
                import train_models
                train_models.train_and_save_models()
                # Clear data cache so dashboard uses full dataset
                load_data.clear()
            except Exception as e:
                st.error(f"Auto-training failed: {e}")
                return None, None, None, None

    try:
        def _load(name):
            with open(f'{MODEL_DIR}/{name}', 'rb') as f:
                return pickle.load(f)
        return (
            _load('cancel_model.pkl'),
            _load('adr_model.pkl'),
            _load('label_encoders.pkl'),
            _load('feature_info.pkl'),
        )
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data(show_spinner=False)
def load_data():
    # Always prefer full dataset over sample — re-check after potential download
    path = _find_data_path()
    if path is None:
        return None
    df = pd.read_csv(path)
    df['children']          = df['children'].fillna(0)
    df['total_guests']      = df['adults'] + df['children'] + df['babies']
    df['total_nights']      = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['arrival_month_num'] = df['arrival_date_month'].map(MONTH_MAP)
    df['adr']               = df['adr'].clip(0, df['adr'].quantile(0.99))
    return df

# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_encode(value, encoder):
    """Encode a value using a saved LabelEncoder; return 0 for unseen labels."""
    try:
        return int(encoder.transform([str(value)])[0])
    except (ValueError, KeyError):
        return 0


def predict_cancellation(inputs, cancel_model, label_encoders, feature_info):
    cancel_features = feature_info['cancel_features']
    row = {
        'hotel_enc'                        : safe_encode(inputs['hotel'],          label_encoders['hotel']),
        'lead_time'                        : inputs['lead_time'],
        'arrival_month_num'                : MONTH_MAP.get(inputs['arrival_month'], 6),
        'stays_in_weekend_nights'          : inputs['weekend_nights'],
        'stays_in_week_nights'             : inputs['week_nights'],
        'total_nights'                     : inputs['weekend_nights'] + inputs['week_nights'],
        'adults'                           : inputs['adults'],
        'children'                         : inputs['children'],
        'babies'                           : inputs['babies'],
        'total_guests'                     : inputs['adults'] + inputs['children'] + inputs['babies'],
        'meal_enc'                         : safe_encode(inputs['meal'],           label_encoders['meal']),
        'market_segment_enc'               : safe_encode(inputs['market_segment'], label_encoders['market_segment']),
        'deposit_type_enc'                 : safe_encode(inputs['deposit_type'],   label_encoders['deposit_type']),
        'customer_type_enc'                : safe_encode(inputs['customer_type'],  label_encoders['customer_type']),
        'is_repeated_guest'                : int(inputs['is_repeated_guest']),
        'previous_cancellations'           : inputs['previous_cancellations'],
        'previous_bookings_not_canceled'   : inputs['previous_bookings_not_canceled'],
        'booking_changes'                  : inputs['booking_changes'],
        'days_in_waiting_list'             : inputs['days_in_waiting_list'],
        'total_of_special_requests'        : inputs['special_requests'],
        'required_car_parking_spaces'      : inputs['parking_spaces'],
        'adr'                              : inputs['adr'],
    }
    X    = pd.DataFrame([row])[cancel_features]
    prob = cancel_model.predict_proba(X)[0][1]
    return float(prob)


def predict_adr(inputs, adr_model, label_encoders, feature_info):
    adr_features = feature_info['adr_features']
    row = {
        'hotel_enc'                   : safe_encode(inputs['hotel'],          label_encoders['hotel']),
        'lead_time'                   : inputs['lead_time'],
        'arrival_month_num'           : MONTH_MAP.get(inputs['arrival_month'], 6),
        'stays_in_weekend_nights'     : inputs['weekend_nights'],
        'stays_in_week_nights'        : inputs['week_nights'],
        'total_nights'                : inputs['weekend_nights'] + inputs['week_nights'],
        'adults'                      : inputs['adults'],
        'children'                    : inputs['children'],
        'babies'                      : inputs['babies'],
        'total_guests'                : inputs['adults'] + inputs['children'] + inputs['babies'],
        'meal_enc'                    : safe_encode(inputs['meal'],              label_encoders['meal']),
        'market_segment_enc'          : safe_encode(inputs['market_segment'],    label_encoders['market_segment']),
        'deposit_type_enc'            : safe_encode(inputs['deposit_type'],      label_encoders['deposit_type']),
        'customer_type_enc'           : safe_encode(inputs['customer_type'],     label_encoders['customer_type']),
        'reserved_room_type_enc'      : safe_encode(inputs['room_type'],         label_encoders['reserved_room_type']),
        'is_repeated_guest'           : int(inputs['is_repeated_guest']),
        'booking_changes'             : inputs['booking_changes'],
        'total_of_special_requests'   : inputs['special_requests'],
        'required_car_parking_spaces' : inputs['parking_spaces'],
    }
    X             = pd.DataFrame([row])[adr_features]
    predicted_adr = adr_model.predict(X)[0]
    return max(0.0, float(predicted_adr))

# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    # ── Global header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🏨 Smart Hotel Intelligence System</h1>
        <p>AI-powered Cancellation Risk Prediction & Revenue Optimisation for Hotel Managers</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load resources ─────────────────────────────────────────────────────────
    cancel_model, adr_model, label_encoders, feature_info = load_models()
    df           = load_data()
    models_ready = cancel_model is not None

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠  Overview",
        "🔮  Cancellation Risk Predictor",
        "💰  Revenue Estimator",
        "📊  Analytics Dashboard",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("### 📌 The Business Problem")
            st.markdown("""
            Hotels worldwide lose **billions of euros** every year because:

            | Problem | Impact |
            |---|---|
            | 📉 Unexpected cancellations | Rooms stay empty — impossible to re-sell last-minute |
            | 💸 Inaccurate revenue forecasting | Pricing decisions made blindly |
            | 🔄 Reactive management | Staff and resources deployed *after* the problem occurs |

            **This system gives hotel managers predictive intelligence** — they see problems
            coming *before* they happen, with zero data science knowledge required.
            """)

            st.markdown("### 🎯 What This System Does")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                <div class="info-card">
                <h4>🔮 Cancellation Risk Predictor</h4>
                Enter any booking's details and instantly receive
                a cancellation probability score (0–100%) with
                colour-coded risk level and specific manager actions.
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown("""
                <div class="info-card">
                <h4>💰 Revenue Estimator</h4>
                Predict the expected nightly room rate (ADR) for any
                booking before confirming, and compare it against
                your hotel's historical average to guide pricing.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🤖 Machine Learning Models")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.info("""
                **Model 1 — Cancellation Predictor**
                - Type: Classification
                - Algorithm: Random Forest (200 trees)
                - Features: 22 booking attributes
                - Output: Probability of cancellation
                """)
            with col_m2:
                st.info("""
                **Model 2 — Revenue Estimator**
                - Type: Regression
                - Algorithm: Random Forest (200 trees)
                - Features: 19 booking attributes
                - Output: Predicted nightly room rate (€)
                """)

        with col_right:
            st.markdown("### 📈 System Performance")

            if models_ready and feature_info:
                stats = feature_info.get('data_stats', {})
                st.metric("Training Dataset",       f"{stats.get('total_records', 0):,} real bookings")
                st.metric("Cancellation Model Accuracy",
                          f"{feature_info.get('cancel_accuracy', 0) * 100:.1f}%")
                st.metric("Revenue Model  R² Score",
                          f"{feature_info.get('adr_r2', 0):.3f}")
                st.metric("Revenue Model  MAE",
                          f"€{feature_info.get('adr_mae', 0):.2f} / night")

                st.markdown("---")
                st.markdown("### 📊 Dataset Highlights")
                st.metric("Overall Cancellation Rate",
                          f"{stats.get('cancel_rate', 0) * 100:.1f}%")
                st.metric("Average Room Rate",
                          f"€{stats.get('avg_adr', 0):.0f} / night")
                st.metric("Average Booking Lead Time",
                          f"{stats.get('avg_lead_time', 0):.0f} days")

            else:
                st.warning("⚠️ Models not found. Train them before using predictions.")
                st.markdown("**Run this command to train:**")
                st.code("python train_models.py", language="bash")

                if st.button("🚀 Train Models Now", type="primary", use_container_width=True):
                    with st.spinner("Training… this takes 2–4 minutes on the first run."):
                        try:
                            import train_models
                            acc, r2, mae = train_models.train_and_save_models()
                            st.success(
                                f"✅ Done!  Accuracy: {acc * 100:.1f}%  |  R²: {r2:.3f}"
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Training failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – CANCELLATION RISK PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        if not models_ready:
            st.error("Models not loaded. Go to the Overview tab and train the models first.")
            st.stop()

        st.markdown("### 🔮 Booking Cancellation Risk Predictor")
        st.caption(
            "Fill in the booking details below. "
            "The AI will instantly score the cancellation risk and recommend actions."
        )

        cat_vals = feature_info.get('cat_unique_values', {})

        with st.form("cancel_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**📋 Booking Details**")
                hotel          = st.selectbox("Hotel Type",
                                    cat_vals.get('hotel', ['City Hotel', 'Resort Hotel']))
                arrival_month  = st.selectbox("Arrival Month", list(MONTH_MAP.keys()), index=5)
                lead_time      = st.number_input("Lead Time (days before arrival)",
                                    min_value=0, max_value=700, value=30)
                deposit_type   = st.selectbox("Deposit Type",
                                    cat_vals.get('deposit_type', ['No Deposit', 'Non Refund', 'Refundable']))
                market_segment = st.selectbox("Market Segment",
                                    cat_vals.get('market_segment',
                                    ['Direct', 'Online TA', 'Offline TA/TO', 'Corporate']))
                customer_type  = st.selectbox("Customer Type",
                                    cat_vals.get('customer_type',
                                    ['Transient', 'Contract', 'Transient-Party', 'Group']))

            with c2:
                st.markdown("**🛏️ Stay Details**")
                weekend_nights  = st.number_input("Weekend Nights",  min_value=0, max_value=20, value=1)
                week_nights     = st.number_input("Week Nights",     min_value=0, max_value=30, value=2)
                meal            = st.selectbox("Meal Plan",
                                    cat_vals.get('meal', ['BB', 'HB', 'FB', 'SC', 'Undefined']))
                adr_input       = st.number_input("Room Rate (€ / night)",
                                    min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
                days_waiting    = st.number_input("Days on Waiting List",
                                    min_value=0, max_value=500, value=0)
                booking_changes = st.number_input("Booking Changes Made",
                                    min_value=0, max_value=20, value=0)

            with c3:
                st.markdown("**👥 Guest Details**")
                adults           = st.number_input("Adults",   min_value=1, max_value=10, value=2)
                children         = st.number_input("Children", min_value=0, max_value=10, value=0)
                babies           = st.number_input("Babies",   min_value=0, max_value=5,  value=0)
                special_requests = st.number_input("Special Requests",       min_value=0, max_value=5, value=0)
                parking_spaces   = st.number_input("Parking Spaces Needed",  min_value=0, max_value=5, value=0)
                is_repeated      = st.checkbox("Returning / Loyal Guest?")
                prev_cancel      = st.number_input("Guest's Previous Cancellations",
                                    min_value=0, max_value=20, value=0)
                prev_bookings    = st.number_input("Guest's Previous Completed Stays",
                                    min_value=0, max_value=50, value=0)

            submitted = st.form_submit_button(
                "🔍 Predict Cancellation Risk", type="primary", use_container_width=True
            )

        if submitted:
            inputs = dict(
                hotel=hotel, lead_time=lead_time, arrival_month=arrival_month,
                weekend_nights=weekend_nights, week_nights=week_nights,
                adults=adults, children=children, babies=babies,
                meal=meal, market_segment=market_segment, deposit_type=deposit_type,
                customer_type=customer_type, is_repeated_guest=is_repeated,
                previous_cancellations=prev_cancel,
                previous_bookings_not_canceled=prev_bookings,
                booking_changes=booking_changes, days_in_waiting_list=days_waiting,
                special_requests=special_requests, parking_spaces=parking_spaces,
                adr=adr_input,
            )

            risk_prob      = predict_cancellation(inputs, cancel_model, label_encoders, feature_info)
            risk_pct       = risk_prob * 100
            total_nights   = weekend_nights + week_nights
            revenue_at_risk = adr_input * total_nights

            # Determine risk tier
            if risk_prob >= 0.65:
                level    = "HIGH RISK"
                css_cls  = "result-high"
                color    = "#ff4d4d"
                emoji    = "🔴"
                actions  = [
                    "Immediately request a **non-refundable deposit** or full pre-payment",
                    "Send a **booking confirmation** reminder to the guest within 24 hours",
                    "Consider **overbooking this room type** by 1 unit to compensate",
                    "Flag this booking for **daily monitoring** until check-in",
                ]
            elif risk_prob >= 0.35:
                level    = "MEDIUM RISK"
                css_cls  = "result-medium"
                color    = "#ffa502"
                emoji    = "🟡"
                actions  = [
                    "Send a **friendly confirmation email** 2 weeks before arrival",
                    "Offer a small **loyalty upgrade** (e.g., room with a view) to cement the booking",
                    "Watch for further **booking changes** — each one increases risk",
                ]
            else:
                level    = "LOW RISK"
                css_cls  = "result-low"
                color    = "#2ed573"
                emoji    = "🟢"
                actions  = [
                    "Standard check-in procedures — no special action needed",
                    "Consider **upselling** premium services (spa, breakfast upgrade)",
                    "This guest is likely to arrive — prepare their room on schedule",
                ]

            st.markdown("---")
            st.markdown("### 📊 Risk Assessment Result")

            col_result, col_gauge = st.columns([2, 1], gap="large")

            with col_result:
                st.markdown(f"""
                <div class="{css_cls}">
                    <div style="font-size: 3.2rem; font-weight: 800; color: {color};">
                        {emoji} {risk_pct:.1f}%
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {color}; margin-top: 0.3rem;">
                        {level}
                    </div>
                    <div style="color: #555; margin-top: 0.4rem;">Cancellation Probability</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.9rem 1.2rem;
                            background: #fff8e1; border-radius: 8px;
                            border-left: 4px solid #ffc107;">
                    <strong>💰 Revenue at Risk:  €{revenue_at_risk:.0f}</strong>
                    &nbsp;·&nbsp;
                    <small>{total_nights} night{'s' if total_nights != 1 else ''} × €{adr_input:.0f}/night</small>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 💡 Recommended Manager Actions")
                for action in actions:
                    st.markdown(f"- {action}")

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Cancellation Risk", 'font': {'size': 13}},
                    number={'suffix': "%", 'font': {'size': 30}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555"},
                        'bar':  {'color': color},
                        'steps': [
                            {'range': [0,  35], 'color': '#d4edda'},
                            {'range': [35, 65], 'color': '#fff3cd'},
                            {'range': [65, 100], 'color': '#f8d7da'},
                        ],
                        'threshold': {
                            'line': {'color': '#333', 'width': 3},
                            'thickness': 0.75,
                            'value': 65,
                        },
                    },
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – REVENUE ESTIMATOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if not models_ready:
            st.error("Models not loaded. Go to the Overview tab and train the models first.")
            st.stop()

        st.markdown("### 💰 Room Revenue Estimator")
        st.caption(
            "Predict the optimal nightly room rate for any booking "
            "and compare it against your hotel's historical average."
        )

        cat_vals = feature_info.get('cat_unique_values', {})

        with st.form("revenue_form"):
            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("**📋 Booking Details**")
                r_hotel   = st.selectbox("Hotel Type",
                                cat_vals.get('hotel', ['City Hotel', 'Resort Hotel']), key='r_h')
                r_month   = st.selectbox("Arrival Month", list(MONTH_MAP.keys()), index=5, key='r_m')
                r_lead    = st.number_input("Lead Time (days)", min_value=0, max_value=700, value=30, key='r_l')
                r_room    = st.selectbox("Room Type",
                                cat_vals.get('reserved_room_type',
                                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']), key='r_rt')
                r_segment = st.selectbox("Market Segment",
                                cat_vals.get('market_segment',
                                ['Direct', 'Online TA', 'Offline TA/TO', 'Corporate']), key='r_s')
                r_deposit = st.selectbox("Deposit Type",
                                cat_vals.get('deposit_type',
                                ['No Deposit', 'Non Refund', 'Refundable']), key='r_d')

            with r2:
                st.markdown("**🛏️ Stay Details**")
                r_weekend = st.number_input("Weekend Nights", min_value=0, max_value=20, value=1, key='r_wk')
                r_week    = st.number_input("Week Nights",    min_value=0, max_value=30, value=2, key='r_wn')
                r_meal    = st.selectbox("Meal Plan",
                                cat_vals.get('meal', ['BB', 'HB', 'FB', 'SC', 'Undefined']), key='r_ml')
                r_ctype   = st.selectbox("Customer Type",
                                cat_vals.get('customer_type',
                                ['Transient', 'Contract', 'Transient-Party', 'Group']), key='r_ct')
                r_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0, key='r_bc')
                r_parking = st.number_input("Parking Spaces", min_value=0, max_value=5,  value=0, key='r_pk')

            with r3:
                st.markdown("**👥 Guest Details**")
                r_adults   = st.number_input("Adults",   min_value=1, max_value=10, value=2, key='r_a')
                r_children = st.number_input("Children", min_value=0, max_value=10, value=0, key='r_c')
                r_babies   = st.number_input("Babies",   min_value=0, max_value=5,  value=0, key='r_b')
                r_special  = st.number_input("Special Requests", min_value=0, max_value=5, value=0, key='r_sp')
                r_repeated = st.checkbox("Returning / Loyal Guest?", key='r_rp')

            r_submitted = st.form_submit_button(
                "💰 Estimate Room Revenue", type="primary", use_container_width=True
            )

        if r_submitted:
            r_inputs = dict(
                hotel=r_hotel, lead_time=r_lead, arrival_month=r_month,
                weekend_nights=r_weekend, week_nights=r_week,
                adults=r_adults, children=r_children, babies=r_babies,
                meal=r_meal, market_segment=r_segment, deposit_type=r_deposit,
                customer_type=r_ctype, room_type=r_room,
                is_repeated_guest=r_repeated, booking_changes=r_changes,
                special_requests=r_special, parking_spaces=r_parking,
            )

            pred_adr     = predict_adr(r_inputs, adr_model, label_encoders, feature_info)
            total_nights = r_weekend + r_week
            total_rev    = pred_adr * total_nights
            avg_adr      = feature_info.get('data_stats', {}).get('avg_adr', 100.0)
            diff_pct     = ((pred_adr - avg_adr) / avg_adr) * 100 if avg_adr else 0

            st.markdown("---")
            st.markdown("### 📊 Revenue Estimate")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Room Rate",   f"€{pred_adr:.2f} / night",
                      delta=f"{diff_pct:+.1f}% vs hotel avg")
            m2.metric("Total Stay Revenue",    f"€{total_rev:.2f}",
                      delta=f"{total_nights} night{'s' if total_nights != 1 else ''}")
            m3.metric("Hotel Historical Avg",  f"€{avg_adr:.0f} / night")

            # Bar chart comparison
            fig_bar = go.Figure()
            fig_bar.add_bar(
                x=['Hotel Average', 'This Booking'],
                y=[avg_adr, pred_adr],
                marker_color=['#74b9ff', '#6c5ce7'],
                text=[f'€{avg_adr:.0f}', f'€{pred_adr:.0f}'],
                textposition='outside',
            )
            fig_bar.update_layout(
                title='Predicted Rate vs. Hotel Historical Average',
                yaxis_title='Average Daily Rate (€)',
                height=320,
                margin=dict(t=50, b=30),
                showlegend=False,
                plot_bgcolor='#fafafa',
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Pricing recommendation
            if pred_adr > avg_adr * 1.1:
                st.success(
                    f"✅ **High-Value Booking** — This booking has above-average revenue potential. "
                    f"Consider a **loyalty discount** to secure it, or **upsell** premium services "
                    f"(e.g., spa, breakfast upgrade)."
                )
            elif pred_adr < avg_adr * 0.9:
                st.warning(
                    f"⚠️ **Below-Average Rate Booking** — Consider adjusting pricing or "
                    f"offering **package deals** (e.g., meal + room) to lift per-booking revenue."
                )
            else:
                st.info(
                    "ℹ️ **Standard Rate Booking** — Revenue is within the normal range. "
                    "Standard operations apply."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – ANALYTICS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if df is None:
            st.error("Dataset not found.")
            st.stop()

        st.markdown("### 📊 Hotel Booking Analytics Dashboard")
        st.caption("Data-driven insights drawn from 119,390 real hotel booking records.")

        # ── KPI row ────────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Bookings",         f"{len(df):,}")
        k2.metric("Cancellation Rate",      f"{df['is_canceled'].mean() * 100:.1f}%")
        k3.metric("Avg Room Rate",          f"€{df['adr'].mean():.0f} / night")
        k4.metric("Avg Booking Lead Time",  f"{df['lead_time'].mean():.0f} days")

        st.markdown("---")

        month_num_to_name = {v: k for k, v in MONTH_MAP.items()}

        # ── Row 1 ──────────────────────────────────────────────────────────────
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            monthly = (
                df.groupby('arrival_month_num')
                  .agg(bookings=('is_canceled', 'count'),
                       cancellations=('is_canceled', 'sum'))
                  .reset_index()
            )
            monthly['cancel_rate'] = monthly['cancellations'] / monthly['bookings'] * 100
            monthly['month_name']  = monthly['arrival_month_num'].map(month_num_to_name)
            monthly = monthly.sort_values('arrival_month_num')

            fig1 = px.bar(
                monthly, x='month_name', y='bookings',
                title='📅 Monthly Booking Volume',
                color='cancel_rate', color_continuous_scale='RdYlGn_r',
                labels={'bookings': 'Total Bookings', 'month_name': 'Month',
                        'cancel_rate': 'Cancel Rate %'},
            )
            fig1.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            seg_cancel = (
                df.groupby('market_segment')['is_canceled']
                  .mean()
                  .reset_index()
                  .rename(columns={'market_segment': 'Market Segment',
                                   'is_canceled': 'Cancellation Rate'})
            )
            seg_cancel['Cancellation Rate'] *= 100
            seg_cancel = seg_cancel.sort_values('Cancellation Rate')

            fig2 = px.bar(
                seg_cancel, x='Cancellation Rate', y='Market Segment', orientation='h',
                title='📦 Cancellation Rate by Market Segment',
                color='Cancellation Rate', color_continuous_scale='RdYlGn_r',
            )
            fig2.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 2 ──────────────────────────────────────────────────────────────
        col3, col4 = st.columns(2, gap="medium")

        with col3:
            adr_trend = (
                df[df['adr'] > 10]
                  .groupby(['hotel', 'arrival_month_num'])['adr']
                  .mean()
                  .reset_index()
            )
            adr_trend['month_name'] = adr_trend['arrival_month_num'].map(month_num_to_name)
            adr_trend = adr_trend.sort_values('arrival_month_num')

            fig3 = px.line(
                adr_trend, x='month_name', y='adr', color='hotel', markers=True,
                title='💶 Average Room Rate by Month & Hotel Type',
                labels={'adr': 'Avg Daily Rate (€)', 'month_name': 'Month'},
            )
            fig3.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            lead_df = df.copy()
            lead_df['lead_bin'] = pd.cut(
                lead_df['lead_time'],
                bins=[0, 7, 14, 30, 60, 90, 180, 365, 700],
                labels=['0–7d', '8–14d', '15–30d', '31–60d',
                        '61–90d', '91–180d', '181–365d', '365d+'],
            )
            lead_summary = (
                lead_df.groupby('lead_bin', observed=True)['is_canceled']
                       .mean()
                       .reset_index()
            )
            lead_summary.columns = ['Lead Time', 'Cancel Rate']
            lead_summary['Cancel Rate'] *= 100

            fig4 = px.bar(
                lead_summary, x='Lead Time', y='Cancel Rate',
                title='⏱️ How Booking Lead Time Affects Cancellation',
                color='Cancel Rate', color_continuous_scale='RdYlGn_r',
                labels={'Cancel Rate': 'Cancellation Rate (%)'},
            )
            fig4.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig4, use_container_width=True)

        # ── Row 3 ──────────────────────────────────────────────────────────────
        col5, col6 = st.columns(2, gap="medium")

        with col5:
            dep_cancel = (
                df.groupby('deposit_type')
                  .agg(bookings=('is_canceled', 'count'),
                       cancel_rate=('is_canceled', 'mean'))
                  .reset_index()
            )
            dep_cancel['cancel_pct'] = dep_cancel['cancel_rate'] * 100

            fig5 = px.bar(
                dep_cancel, x='deposit_type', y='cancel_pct',
                title='💳 Cancellation Rate by Deposit Type',
                color='cancel_pct', color_continuous_scale='RdYlGn_r',
                labels={'deposit_type': 'Deposit Type', 'cancel_pct': 'Cancellation Rate (%)'},
            )
            fig5.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            if models_ready and feature_info:
                imp_raw = feature_info.get('cancel_importances', {})
                name_map = {
                    'lead_time': 'Lead Time',
                    'adr': 'Room Rate',
                    'arrival_month_num': 'Arrival Month',
                    'total_nights': 'Total Nights',
                    'deposit_type_enc': 'Deposit Type',
                    'market_segment_enc': 'Market Segment',
                    'previous_cancellations': 'Previous Cancellations',
                    'total_of_special_requests': 'Special Requests',
                    'days_in_waiting_list': 'Days on Waiting List',
                    'booking_changes': 'Booking Changes',
                    'customer_type_enc': 'Customer Type',
                    'total_guests': 'Total Guests',
                    'is_repeated_guest': 'Returning Guest',
                    'hotel_enc': 'Hotel Type',
                    'meal_enc': 'Meal Plan',
                }
                imp_df = (
                    pd.DataFrame({'Feature': list(imp_raw.keys()),
                                  'Importance': list(imp_raw.values())})
                    .assign(Feature=lambda d: d['Feature'].map(
                        lambda x: name_map.get(x, x.replace('_enc', '').replace('_', ' ').title())
                    ))
                    .sort_values('Importance')
                    .tail(12)
                )

                fig6 = px.bar(
                    imp_df, x='Importance', y='Feature', orientation='h',
                    title='🔑 Top Cancellation Risk Factors (AI Feature Importance)',
                    color='Importance', color_continuous_scale='Blues',
                )
                fig6.update_layout(height=320, margin=dict(t=50, b=30))
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Train the models to see AI feature importance chart.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.82rem; padding: 0.4rem;">
        🏨 Smart Hotel Intelligence System &nbsp;|&nbsp;
        University Final Year Project &nbsp;|&nbsp;
        Built with Python · Scikit-learn · Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
