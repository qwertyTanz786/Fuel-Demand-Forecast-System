import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
st.markdown("""
<style>

/* =========================================
   MAIN BACKGROUND
========================================= */

/* =====================================
   GOOGLE FONT
===================================== */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* =====================================
   ANIMATED BACKGROUND
===================================== */

.stApp{

    background:linear-gradient(
        -45deg,
        #0f172a,
        #1e293b,
        #334155,
        #0f172a
    );

    background-size:400% 400%;

    animation:bgMove 15s ease infinite;

    color:white;
}

@keyframes bgMove{

    0%{
        background-position:0% 50%;
    }

    50%{
        background-position:100% 50%;
    }

    100%{
        background-position:0% 50%;
    }
}

/* =====================================
   DYNAMIC DEMAND OVERLAY
===================================== */

.high-demand{

    position:fixed;

    top:0;
    left:0;

    width:100%;
    height:100%;

    background:
    radial-gradient(
        circle,
        rgba(255,0,0,.18),
        transparent 70%
    );

    backdrop-filter:blur(25px);

    pointer-events:none;

    animation:pulseRed 3s infinite;
}

.medium-demand{

    position:fixed;

    top:0;
    left:0;

    width:100%;
    height:100%;

    background:
    radial-gradient(
        circle,
        rgba(255,200,0,.15),
        transparent 70%
    );

    backdrop-filter:blur(15px);

    pointer-events:none;
}

.low-demand{

    position:fixed;

    top:0;
    left:0;

    width:100%;
    height:100%;

    background:
    radial-gradient(
        circle,
        rgba(0,255,100,.12),
        transparent 70%
    );

    backdrop-filter:blur(10px);

    pointer-events:none;
}

@keyframes pulseRed{

    0%{
        opacity:.4;
    }

    50%{
        opacity:1;
    }

    100%{
        opacity:.4;
    }
}

/* =====================================
   TITLE
===================================== */

/* =====================================
   TITLE
===================================== */

.main-title{

    text-align:center;

    font-size:4rem;

    font-weight:800;

    color:white;

    letter-spacing:2px;

    margin-bottom:10px;
}

@keyframes glowTitle{

    from{
        text-shadow:
        0 0 15px rgba(249,115,22,.4);
    }

    to{
        text-shadow:
        0 0 40px rgba(249,115,22,.9);
    }
}

/* =====================================
   SIDEBAR
===================================== */

[data-testid="stSidebar"]{

    background:
    rgba(15,23,42,.75);

    backdrop-filter:blur(25px);

    border-right:
    1px solid rgba(255,255,255,.08);

    box-shadow:
    0 0 40px rgba(0,0,0,.4);
}

/* =====================================
   GLASS CARDS
===================================== */

.card,
[data-testid="metric-container"]{

    background:
    rgba(255,255,255,.08);

    backdrop-filter:
    blur(25px);

    border:
    1px solid rgba(255,255,255,.12);

    border-radius:25px;

    transition:all .4s ease;

    box-shadow:
    0 8px 32px rgba(0,0,0,.2);
}

.card:hover,
[data-testid="metric-container"]:hover{

    transform:
    translateY(-10px)
    scale(1.03);

    box-shadow:
    0 0 35px rgba(249,115,22,.4);
}

/* =====================================
   DROPDOWNS
===================================== */

.stSelectbox{

    transition:all .3s ease;
}

.stSelectbox:hover{

    transform:translateY(-3px);

    filter:
    drop-shadow(
        0 0 15px rgba(249,115,22,.5)
    );
}

/* =====================================
   BUTTON
===================================== */

.stButton > button{

    width:100%;

    height:65px;

    border:none;

    border-radius:20px;

    background:
    linear-gradient(
        135deg,
        #f97316,
        #fb923c
    );

    color:white;

    font-weight:700;

    font-size:18px;

    transition:all .3s ease;

    box-shadow:
    0 0 25px rgba(249,115,22,.4);
}

.stButton > button:hover{

    transform:
    scale(1.05);

    box-shadow:
    0 0 50px rgba(249,115,22,.8);
}

.stButton > button:active{

    transform:scale(.95);
}

/* =====================================
   METRIC VALUE ANIMATION
===================================== */

[data-testid="stMetricValue"]{

    animation:
    pulseMetric 2s infinite;
}

@keyframes pulseMetric{

    0%{
        transform:scale(1);
    }

    50%{
        transform:scale(1.08);
    }

    100%{
        transform:scale(1);
    }
}

/* =====================================
   PLOTLY CHART
===================================== */

[data-testid="stPlotlyChart"]{

    background:Black;

    backdrop-filter:
    blur(25px);

    border-radius:25px;

    padding:20px;

    transition:all .4s ease;
}

[data-testid="stPlotlyChart"]:hover{

    transform:
    scale(1.02);

    box-shadow:
    0 0 40px rgba(249,115,22,.35);
}

/* =====================================
   FADE IN
===================================== */

.block-container{

    animation:
    fadeInUp .8s ease;
}

@keyframes fadeInUp{

    from{
        opacity:0;
        transform:translateY(25px);
    }

    to{
        opacity:1;
        transform:translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("fuel_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# Hero Section
st.markdown("""
<div class='main-title'>
⛽ UAE Smart Fuel Demand Forecasting System
</div>
""", unsafe_allow_html=True)

st.markdown(
"""
<center style='color:silver'>
AI-Powered Fuel Analytics Dashboard
</center>
""",
unsafe_allow_html=True
)

st.write("")

# Sidebar Header
st.sidebar.title("🚗 Vehicle Configuration")

tank_map = {
  # Sedans
    "Toyota Corolla": 50,
    "Toyota Camry": 60,
    "Honda Accord": 56,
    "Hyundai Elantra": 50,
    "Nissan Altima": 62,

    # SUVs
    "Toyota Land Cruiser": 110,
    "Nissan Patrol": 140,
    "Toyota Prado": 87,
    "Ford Explorer": 80,
    "Kia Sportage": 62,
    "Bentley Bentayga": 85,
    "Rolls Royce Cullinan": 90,
    "Mercedes G63 AMG": 100,
    "Range Rover SV": 90,
    "Lamborghini Urus": 85,
    "Brabus G800": 100,
    "Hyundai Tucson": 54,
    "Nissan X-Trail": 55,
    "Toyota RAV4": 55,
    "Chevrolet Tahoe": 91,
    "BMW X5": 83,
    "Mercedes GLE": 85,
    "Audi Q8": 85,

    # Sports Cars / Supercars / Hypercars
    "Ferrari 488": 78,
    "Ferrari F8 Tributo": 78,
    "Ferrari SF90 Stradale": 68,
    "Ferrari LaFerrari": 72,

    "Lamborghini Huracan": 83,
    "Lamborghini Aventador SVJ": 85,

    "McLaren 720S": 72,
    "McLaren 765LT": 72,
    "McLaren P1": 72,

    "Porsche 911 Turbo": 67,
    "Porsche 918 Spyder": 70,

    "Audi R8": 73,
    "Nissan GT-R": 74,
    "Mercedes AMG GT R": 75,

    "Bugatti Chiron": 100,
    "Bugatti Veyron": 100,

    "Koenigsegg Jesko": 80,

    "Pagani Huayra": 85,
    "Pagani Utopia": 80,

    "Aston Martin Valkyrie": 70,

    # Taxis
    "Toyota Camry Taxi": 60,
    "Lexus ES Taxi": 65,
    "Hyundai Sonata Taxi": 60,

    # Trucks
    "Mercedes Actros": 500,
    "Volvo FH": 600,
    "MAN TGX": 550,
    "Scania R450": 650,

    # Buses
    "Toyota Coaster": 95,
    "Mercedes Sprinter": 93,
    "Ashok Leyland Falcon": 180,

    # Delivery Vans
    "Toyota Hiace": 70,
    "Ford Transit": 80,
    "Nissan Urvan": 65,
}

fuel_rules = {
    "Sedan": ["Special 95", "Super 98", "E-Plus 91"],
    "SUV": ["Special 95", "Super 98", "E-Plus 91"],
    "Sports Car": ["Super 98"],
    "Truck": ["Diesel"],
    "Bus": ["Diesel"],
    "Taxi": ["Special 95", "E-Plus 91"],
    "Delivery Van": ["Diesel"]
}

def get_fuel_price(fuel_type):
    prices = {
        "Diesel": 4.33,
        "E-Plus 91": 3.76,
        "Special 95": 3.83,
        "Super 98": 3.95
    }
    return prices.get(fuel_type, 0)

vehicle_types={
    "Sedan": ["Toyota Corolla", "Toyota Camry", "Honda Accord", "Hyundai Elantra", "Nissan Altima"],
    "SUV": ["Toyota Land Cruiser", "Nissan Patrol", "Toyota Prado", "Ford Explorer", "Kia Sportage",
            "Bentley Bentayga", "Rolls Royce Cullinan", "Mercedes G63 AMG", "Range Rover SV",
            "Lamborghini Urus", "Brabus G800","Hyundai Tucson","Nissan X-Trail","Toyota RAV4","Chevrolet Tahoe","BMW X5","Mercedes GLE",
            "Audi Q8"],
    "Sports Car": ["Ferrari 488", "Ferrari F8 Tributo", "Ferrari SF90 Stradale", "Ferrari LaFerrari",
                    "Lamborghini Huracan", "Lamborghini Aventador SVJ",
                    "McLaren 720S", "McLaren 765LT", "McLaren P1",
                    "Porsche 911 Turbo", "Porsche 918 Spyder",
                    "Audi R8", "Nissan GT-R", "Mercedes AMG GT R",
                    "Bugatti Chiron", "Bugatti Veyron",
                    "Koenigsegg Jesko",
                    "Pagani Huayra", "Pagani Utopia",
                    "Aston Martin Valkyrie"],
    "Truck": ["Mercedes Actros", "Volvo FH", "MAN TGX", "Scania R450"],
    "Bus": ["Toyota Coaster", "Mercedes Sprinter", "Ashok Leyland Falcon"],
    "Taxi": ["Toyota Camry Taxi", "Lexus ES Taxi", "Hyundai Sonata Taxi"],
    "Delivery Van": ["Toyota Hiace", "Ford Transit", "Nissan Urvan"]        
}

class_map = {
    "Sedan":"Light",
    "SUV":"Medium",
    "Sports Car":"Performance",
    "Truck":"Heavy",
    "Bus":"Heavy",
    "Taxi":"Commercial",
    "Delivery Van":"Commercial"
}

traffic_map = {
    "Dubai":5,
    "Abu Dhabi":4,
    "Sharjah":3,
    "Ajman":2,
    "Al Ain":2
}

vehicle_model = st.sidebar.selectbox(
    "Vehicle Model",
    options=list(tank_map.keys())
)

emirate = st.sidebar.selectbox(
    "Emirate",
    options=list(traffic_map.keys())
)

month = st.sidebar.selectbox(
    "Month",
    list(range(1,13))
)

weekday = st.sidebar.selectbox(
    "Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

weekday_num = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
].index(weekday)

petrol_station = st.sidebar.selectbox(
    "Petrol Station",
    ["ENOC","EPPCO","ADNOC","CAFU","Emarat","All Companies"]
)
selected_type = None

for vtype, models in vehicle_types.items():
    if vehicle_model in models:
        selected_type = vtype
        break

Vehicle_Type = selected_type
st.sidebar.success(
    f"Vehicle Type: {Vehicle_Type}"
)
allowed_fuels = fuel_rules[Vehicle_Type]

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    allowed_fuels
)
if st.button("Predict Fuel Demand"):
    prediction_station = petrol_station
    if petrol_station == "All Companies":
        prediction_station = "ENOC"
    user_data = {
        "Fuel_Type": fuel_type,
        "Vehicle_Model": vehicle_model,
        "Vehicle_Type": Vehicle_Type,
        "Emirate": emirate,
        "Petrol_Station": prediction_station,
        "Month": month,
        "Weekday": weekday_num
    }
    user_data["Tank_Capacity"] = tank_map[vehicle_model]
    user_data["Vehicle_Class"] = class_map[Vehicle_Type]
    user_data["Traffic_Score"] = traffic_map[emirate]
    user_data["Station_Popularity"] = 1000
    user_df = pd.DataFrame([user_data])

    user_df = pd.get_dummies(user_df)

    user_df = user_df.reindex(
        columns=training_columns,
        fill_value=0
    )

    prediction = model.predict(user_df)

    fuel_price = get_fuel_price(fuel_type)

    estimated_cost = prediction[0] * fuel_price

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fuel Demand",
            f"{prediction[0]:.2f} L"
        )

    with col2:
        st.metric(
            "Estimated Cost",
            f"{estimated_cost:.2f} AED"
        )

    with col3:
        st.metric(
            "Tank Capacity",
            f"{tank_map[vehicle_model]} L"
        )
    st.subheader("📊 Fuel Demand Comparison")

    companies = ["ENOC", "EPPCO", "ADNOC", "CAFU", "Emarat"]
    graph_data = []

    for company in companies:

        temp_data = {
            "Fuel_Type": fuel_type,
            "Vehicle_Model": vehicle_model,
            "Vehicle_Type": Vehicle_Type,
            "Emirate": emirate,
            "Petrol_Station": company,
            "Month": month,
            "Weekday": weekday_num,
            "Tank_Capacity": tank_map[vehicle_model],
            "Vehicle_Class": class_map[Vehicle_Type],
            "Traffic_Score": traffic_map[emirate],
            "Station_Popularity": 1000
        }

        temp_df = pd.DataFrame([temp_data])

        temp_df = pd.get_dummies(temp_df)

        temp_df = temp_df.reindex(
            columns=training_columns,
            fill_value=0
        )

        company_prediction = model.predict(temp_df)[0]

        graph_data.append({
            "Company": company,
            "Fuel Demand": round(company_prediction, 2)
        })

        graph_df = pd.DataFrame(graph_data)

# ALL COMPANIES VIEW
    if petrol_station == "All Companies":

        fig = px.bar(
            graph_df,
            x="Company",
            y="Fuel Demand",
            text="Fuel Demand",
            title="Fuel Demand Forecast Across All Petrol Companies"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=500,
            xaxis_title="Petrol Company",
            yaxis_title="Fuel Demand (Litres)"
        )

        st.plotly_chart(fig, use_container_width=True)

# SINGLE COMPANY VIEW
    else:

        company_row = graph_df[
            graph_df["Company"] == petrol_station
        ]

        fig = px.bar(
            company_row,
            x="Company",
            y="Fuel Demand",
            text="Fuel Demand",
            title=f"Fuel Demand Forecast - {petrol_station}"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=450,
            xaxis_title="Petrol Company",
            yaxis_title="Fuel Demand (Litres)"
        )

        st.plotly_chart(fig, use_container_width=True)

        fill_percent = min(
            prediction[0] / tank_map[vehicle_model],
            1.0
        )

        st.subheader("Tank Utilization")

        st.progress(int(fill_percent * 100))

        if prediction[0] <= (
            0.33 * tank_map[vehicle_model]
        ):
            st.info(
                "Fuel demand is low. Your current tank should be sufficient."
            )

        if prediction[0] > 100:
            st.warning(
                "High fuel demand forecast."
            )

        if prediction[0] > tank_map[vehicle_model]:
            st.error(
                "Forecast exceeds vehicle tank capacity."
            )
    