from calendar import Month
import joblib
from difflib import get_close_matches
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"C:\Portfolio-ML\Datasets\UAE_Fuel_Dataset_Monthly_Pricing.csv")
print("="*60)
print("Smart Fuel Demand Forecasting System")
print("="*60)
print("""
This tool predicts expected fuel demand based on:

✓ Vehicle Type
✓ Vehicle Model
✓ Fuel Type
✓ Emirate
✓ Petrol Station
✓ Date Information

Let's get started.
Loading dataset and preparing the model...
""")
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
df["Tank_Capacity"] = df["Vehicle_Model"].map(tank_map)
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
# Define vehicle types and their corresponding models
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
df["Vehicle_Class"] = df["Vehicle_Type"].map(class_map)

traffic_map = {
    "Dubai":5,
    "Abu Dhabi":4,
    "Sharjah":3,
    "Ajman":2,
    "Al Ain":2
}
df["Traffic_Score"] = df["Emirate"].map(traffic_map)
station_counts = df["Petrol_Station"].value_counts()
df["Station_Popularity"] = df["Petrol_Station"].map(station_counts)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday
df.drop('Date', axis=1, inplace=True)
df=pd.get_dummies(df,drop_first=True)
X=df.drop(['Fuel_Demand_Liters', 'Fuel_Price_AED'], axis=1)
Y=df['Fuel_Demand_Liters']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
training_columns = X.columns
m=XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
m.fit(X_train,Y_train)
predictions=m.predict(X_test)
#Emirate validation
while True:
    emirate = input("Enter Emirate (Dubai, Abu Dhabi, Sharjah, Ajman, Al Ain): ")
    if emirate in traffic_map:
        break
    suggestion = get_close_matches(emirate, traffic_map.keys(), n=1)
    if suggestion:
        print(f"Did you mean '{suggestion[0]}'?")
        if input("Y or N? ").strip().lower() == "y":
            emirate = suggestion[0]
            break
    print("Invalid Emirate.")

while True: 
    try: 
        month = int(input("Enter Current Month (1-12): ")) 
        if 1 <= month <= 12: 
            break 
        print("❌ Invalid month. Please enter a number between 1 and 12.") 
    except ValueError: print("❌ Please enter numbers only.")
#Weekday Validation
while True:
    try:
        weekday = int(input("Enter Current Day (0=Mon, 6=Sun): "))
        if 0 <= weekday <= 6:
            break
        print("❌ Invalid day. Please enter a number between 0 and 6.")
    except ValueError:
        print("❌ Please enter numbers only.")
#Vehicle Model Validation
print("\nAvailable Vehicle Models:")
for model in tank_map.keys():
    print(f"• {model}")
while True:
    vehicle_model = input("\nEnter Vehicle Model: ").strip()
    if vehicle_model in tank_map:
        break
    suggestion = get_close_matches(vehicle_model, tank_map.keys(), n=1)
    if suggestion:
        print(f"❌ Model not found. Did you mean '{suggestion[0]}'?")
        print("Y or N?")
        if input().strip().lower() == 'y':
            vehicle_model = suggestion[0]
            break
    else:
        print("❌ Invalid vehicle model. Please choose from the list above.")
selected_type = None
for vtype, models in vehicle_types.items():
    if vehicle_model in models:
        selected_type = vtype
        break
Vehicle_Type = selected_type
print(f"\nDetected Vehicle Type: {Vehicle_Type}")
allowed_fuels = fuel_rules[Vehicle_Type]
#Vehicle Type Validation
print("\nAllowed Fuel Types:")
for fuel in allowed_fuels:
    print(f"• {fuel}")
while True:
    fuel_type = input("\nEnter Fuel Type: ").strip()
    if fuel_type in allowed_fuels:
        break
    suggestion = get_close_matches(
        fuel_type,
        allowed_fuels,
        n=1
    )
    if suggestion:
        print(
            f"❌ Invalid fuel for {Vehicle_Type}. "
            f"Did you mean '{suggestion[0]}'?"
        )
        if input("Y or N? ").lower() == "y":
            fuel_type = suggestion[0]
            break
    else:
        print(
            f"❌ {Vehicle_Type} only supports: "
            f"{', '.join(allowed_fuels)}"
        )
#Petrol Station Validation
while True:
    petrol_station = input("Petrol Station Company (ENOC, EPPCO, ADNOC, CAFU, Emarat): ").strip().upper()
    if petrol_station in {"ENOC", "EPPCO", "ADNOC", "CAFU", "Emarat"}:
        break
    suggestion = get_close_matches(petrol_station, {"ENOC", "EPPCO", "ADNOC", "CAFU", "Emarat"}, n=1)
    if suggestion:
        print(f"❌ Station not found. Did you mean '{suggestion[0]}'?")
        print("Y or N?")
        if input().strip().lower() == 'y':
            petrol_station = suggestion[0]
            break
    print("❌ Invalid petrol station. Please enter a valid station.")

user_data = {
    "Fuel_Type": fuel_type,
    "Vehicle_Model": vehicle_model,
    "Vehicle_Type": Vehicle_Type,
    "Emirate": emirate,
    "Petrol_Station": petrol_station,
    "Month": month,
    "Weekday": weekday
}
user_data["Tank_Capacity"] = tank_map.get(
    user_data["Vehicle_Model"],
)
user_data["Vehicle_Class"] = class_map.get(user_data["Vehicle_Type"])
user_data["Traffic_Score"] = traffic_map.get(user_data["Emirate"])
user_data["Station_Popularity"] = station_counts.get(
    user_data["Petrol_Station"],
    0
)
user_df = pd.DataFrame([user_data])
user_df = pd.get_dummies(user_df)
user_df = user_df.reindex(
    columns=training_columns,
    fill_value=0
)
prediction = m.predict(user_df)
if prediction[0] <= (0.33 * user_data["Tank_Capacity"]):
    print("""\n⚠️ Fuel demand forecast is low. Your vehicle's tank should suffice for your needs.
          Consider refueling at your convenience.""")
if prediction[0]>user_data["Tank_Capacity"]:
    print("\n⚠️  Warning: Forecasted fuel demand exceeds your vehicle's tank capacity!")
if prediction[0]>100:
    print("⚠️  Note: High fuel demand forecast may indicate increased driving or heavy traffic conditions.")
print("\n" + "="*50)
print("FUEL DEMAND FORECAST")
print("="*50)
print(f"Vehicle Model : {vehicle_model}")
print(f"Vehicle Type  : {Vehicle_Type}")
print(f"Fuel Type     : {fuel_type}")
print(f"Emirate       : {emirate}")
print(f"Month         : {Month(month).name}")
print(f"Day of Week   : {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][weekday]}")
print(f"Station       : {petrol_station}")
print("-"*50)
print("""
      
      *Disclaimer:The fuel prices may vary with the current market price of the fuel types
      and the estimated cost is derived from the forecasted demand and the current fuel price.
      The actual fuel demand and cost may differ based on various factors such as driving habits,
      traffic conditions, and fuel efficiency of the vehicle.
      This forecast is an estimate and should be used for informational purposes only.
      Always consider real-time data and consult with local fuel providers for accurate pricing.
      
      """)
print(f"Forecasted Fuel Demand for Your Vehicle : {prediction[0]:,.2f} Liters")
fuel_price = get_fuel_price(fuel_type)
estimated_cost = prediction[0] * fuel_price
print(f"Estimated Fuel Cost (Derived from Forecasted Demand)    : {estimated_cost:,.2f} AED")
print("="*50)
joblib.dump(m, "fuel_model.pkl")
joblib.dump(training_columns, "training_columns.pkl")