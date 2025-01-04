#%% Import Libraries
import pandas as pd
import warnings
import numpy as np

#%% Load the dataset and merge it
data_url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
avocado = pd.read_csv(data_url+"HAB_data_2015to2022.csv")
avocado["date"] = pd.to_datetime(avocado["date"])
avocado = avocado.sort_values(by="date")
print(avocado)

#%% removing the Total_US rregion in the regions column
regions = [
    "Great_Lakes",
    "Midsouth",
    "Northeast",
    "Northern_New_England",
    "SouthCentral",
    "Southeast",
    "West",
    "Plains"
]
df = avocado[avocado.region.isin(regions)]

#%% Train the model to predict total units sold
from sklearn.model_selection import train_test_split

X = df[["region", "price", "year", "peak"]]
y = df["units_sold"]
# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1
)

#%% categorical encoding needed for column REGION
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

feat_transform = make_column_transformer(
    (OneHotEncoder(drop="first"), ["region"]),
    (StandardScaler(), ["price", "year"]),
    ("passthrough", ["peak"]),
    verbose_feature_names_out=False,
    remainder='drop'
)

#%%
reg = make_pipeline(feat_transform, LinearRegression())
reg.fit(X_train, y_train)

# Get R^2 from test data
y_pred = reg.predict(X_test)
print(f"The R^2 value in the test set is {np.round(r2_score(y_test, y_pred),5)}")

#%%
reg.fit(X, y)

y_pred_full = reg.predict(X)
print(f"The R^2 value in the full dataset is {np.round(r2_score(y, y_pred_full),5)}")

#%% PART-B: Optimize for Price and Supply of Avocados
# Sets and parameters
B = 35  # total amount of avocado supply

peak_or_not = 1  # 1 if it is the peak season; 0 if isn't
year = 2022

c_waste = 0.1  # the cost ($) of wasting an avocado

# the cost of transporting an avocado
c_transport = pd.Series(
    {
        "Great_Lakes": 0.3,
        "Midsouth": 0.1,
        "Northeast": 0.4,
        "Northern_New_England": 0.5,
        "SouthCentral": 0.3,
        "Southeast": 0.2,
        "West": 0.2,
        "Plains": 0.2,
    }, name='transport_cost'
)
c_transport = c_transport.loc[regions]

a_min = 0  # minimum avocado price
a_max = 3  # maximum avocado price

# Get the lower and upper bounds from the dataset for the price and the number of products to be stocked
data = pd.concat([c_transport,
                  df.groupby("region")["units_sold"].min().rename('min_delivery'),
                  df.groupby("region")["units_sold"].max().rename('max_delivery')], axis=1)

print(data)

#%% Import Gurobipy packages
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr

#%% Create fixed features
feats = pd.DataFrame(
    data={
        "year": year,
        "peak": peak_or_not,
        "region": regions,
    },
    index=regions
)
print(feats)

#%% Add Decision Variables
import gurobipy as gp

m = gp.Model("Avocado_Price_Allocation")

p = gppd.add_vars(m, data, name="price", lb=a_min, ub=a_max) # price of an avocado for each region
x = gppd.add_vars(m, data, name="x", lb='min_delivery', ub='max_delivery') # number of avocados supplied to each reagion
s = gppd.add_vars(m, data, name="s") # predicted amount of sales in each region for the given price
w = gppd.add_vars(m, data, name="w") # excess wasteage in each region
d = gppd.add_vars(m, data, lb=-gp.GRB.INFINITY, name="demand") # Add variables for the regression

m.update()

#%% Add the supply constraint for optimization
m.addConstr(x.sum() == B)
m.update()

#%% Add Constraints That Define Sales Quantity
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, x)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, d)
m.update()

#%% Wastage constraint
gppd.add_constrs(m, w, gp.GRB.EQUAL, x - s)
m.update()

#%% add all object variables to predict demand
m_feats = pd.concat([feats, p], axis=1)[["region", "price", "year", "peak"]]
print(m_feats)

#%% Insert the constraints linking the features and the demand into the model m
pred_constr = add_predictor_constr(m, reg, m_feats, d)
print(pred_constr.print_stats())

#%%
m

#%% Set the objective for optimization
m.setObjective((p * s).sum() - c_waste * w.sum() - (c_transport * x).sum(), gp.GRB.MAXIMIZE)

#%%
m.Params.NonConvex = 2
m.optimize()

#%%
solution = pd.DataFrame(index=regions)

solution["Price"] = p.gppd.X
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = w.gppd.X
solution["Pred_demand"] = d.gppd.X

opt_revenue = m.ObjVal
print("\n The optimal net revenue: $%f million" % opt_revenue)
solution.round(4)

#%%
print(solution)


#%%check error too
print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)

#%%Start saving and loading model
import joblib

# Save the regression model
joblib.dump(reg, "regression_model.joblib")
# Load the regression model
reg = joblib.load("regression_model.joblib")

#%%Built streamlit interface with sliders and output

# Define regions and transport costs
regions = [
    "Great_Lakes", "Midsouth", "Northeast", "Northern_New_England",
    "SouthCentral", "Southeast", "West", "Plains"
]
transport_cost = {
    "Great_Lakes": 0.3, "Midsouth": 0.1, "Northeast": 0.4,
    "Northern_New_England": 0.5, "SouthCentral": 0.3,
    "Southeast": 0.2, "West": 0.2, "Plains": 0.2,
}

#%% Optimization results (replace with dynamic integration if needed)
optimization_results = {
    "Price": [1.613872, 1.508809, 1.989001, 1.441157, 2.002708, 1.696370, 2.154178, 1.152070],
    "Allocated": [3.556603, 6.168572, 4.162924, 0.917984, 4.413509, 5.532812, 7.081995, 3.165602],
    "Sold": [3.556603, 3.545445, 4.162924, 0.917984, 4.413509, 3.958786, 4.967691, 2.759275],
    "Wasted": [0.0, 2.623127, 0.0, 0.0, 0.0, 1.574025, 2.114304, 0.406327],
    "Predicted Demand": [3.556603, 3.545445, 4.162924, 0.917984, 4.413509, 3.958786, 4.967691, 2.759275]
}
opt_revenue = 41.167116  # Optimal revenue value from the model
max_error = 1.33227e-15  # Maximum prediction error

#%% Streamlit interface
import streamlit as st
st.title("Avocado Price and Supply Optimization")

# User input for prediction
st.header("Predict Units Sold")
region = st.selectbox("Select Region", regions)
price = st.slider("Set Avocado Price ($)", min_value=0.0, max_value=3.0, step=0.1)
year = st.slider("Select Year", min_value=2015, max_value=2023, step=1)
peak = st.radio("Is it peak season?", [1, 0], format_func=lambda x: "Yes" if x else "No")

# Prediction
input_features = pd.DataFrame({"region": [region], "price": [price], "year": [year], "peak": [peak]})
predicted_units = reg.predict(input_features)[0]
st.write(f"Predicted Units Sold: {int(predicted_units)}")

# Display Optimization Results
st.header("Optimization Results")
if st.button("Run Optimization"):
    solution = pd.DataFrame(optimization_results, index=regions)
    solution.index.name = "Region"

    st.write(f"Optimal Net Revenue: ${opt_revenue:.6f} million")
    st.dataframe(solution)

    # Visualization
    st.bar_chart(solution[["Sold", "Allocated", "Wasted"]])

# Add error check
st.header("Model Error Check")
if st.checkbox("Show Maximum Prediction Error"):
    st.write(f"Maximum Error in Approximation: {max_error:.6f}")
