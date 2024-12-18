import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
data = pd.read_excel("Data_Train.xlsx")  # Replace "flights.xlsx" with your actual file name

# Step 1: Feature Engineering
# Split Date_of_Journey
data["Journey_day"] = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.day
data["Journey_month"] = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.month
data.drop("Date_of_Journey", axis=1, inplace=True)

# Split Dep_Time
data["Dep_hour"] = pd.to_datetime(data["Dep_Time"]).dt.hour
data["Dep_min"] = pd.to_datetime(data["Dep_Time"]).dt.minute
data.drop("Dep_Time", axis=1, inplace=True)

# Split Arrival_Time
data["Arrival_hour"] = pd.to_datetime(data["Arrival_Time"]).dt.hour
data["Arrival_min"] = pd.to_datetime(data["Arrival_Time"]).dt.minute
data.drop("Arrival_Time", axis=1, inplace=True)

# Convert Duration to hours and minutes
data['Duration_hours'] = data['Duration'].apply(lambda x: int(x.split()[0][0:-1]) if 'h' in x else 0)
data['Duration_mins'] = data['Duration'].apply(lambda x: int(x.split()[-1][0:-1]) if 'm' in x else 0)
data.drop("Duration", axis=1, inplace=True)

# Convert Total_Stops
data["Total_Stops"] = data["Total_Stops"].map({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4})

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['Airline', 'Source', 'Destination', 'Additional_Info']:
    data[col] = label_encoder.fit_transform(data[col])

# Drop irrelevant columns
data.drop(columns=['Route'], inplace=True)

# Step 2: Split data into features and target
X = data.drop(columns=["Price"])
y = data["Price"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")

# Save the model for later use
joblib.dump(model, 'flight_fare_predictor.pkl')
