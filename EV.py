#!/usr/bin/env python
# coding: utf-8

# # 🚗 Electric Vehicle Population Data Analysis

# ## 1️⃣ Project Overview
# Electric Vehicles (EVs) are becoming increasingly popular as a cleaner alternative to traditional fuel-based vehicles. This project analyzes a dataset of registered EVs to understand trends in EV adoption, key manufacturers, and other relevant insights.
# 

# ## 2️⃣ Objectives
# - 📆 **EV Adoption Over Time**: Analyze the growth of the EV population by model year.
# - 🗺️ **Geographical Distribution**: Understand where EVs are most commonly registered (e.g., by county or city).
# - 🔋 **EV Types**: Breakdown of the dataset by electric vehicle type (BEV, etc.).
# - 🚘 **Make and Model Popularity**: Identify the most popular makes and models among the registered EVs.
# - ⚡ **Electric Range Analysis**: Analyze the electric range of vehicles to see how EV technology is progressing.
# - 📈 **Estimated Growth in Market Size**: Analyze and find the estimated growth in the market size of electric vehicles.

# ## 3️⃣ Dataset Overview & Loading

# ### 📌 Source
# 
# This dataset is sourced from the U.S. government’s open data portal ([data.gov](https://www.data.gov/)) and contains information on registered electric vehicles.
# 
# 
# 

# ### 📋 Description
# The dataset includes records of registered EVs with the following key features:
# 
# - **VIN (1-10):** First 10 characters of the Vehicle Identification Number (VIN).
# - **County:** County where the vehicle is registered.
# - **City:** City where the vehicle is registered.
# - **State:** State of registration.
# - **Postal Code:** ZIP code of the registered location.
# - **Model Year:** Manufacturing year of the vehicle.
# - **Make:** Vehicle manufacturer (e.g., Tesla, Nissan, BMW).
# - **Model:** Specific car model.
# - **Electric Vehicle Type:** BEV (Battery Electric Vehicle) or PHEV (Plug-in Hybrid).
# - **CAFV Eligibility:** Indicates if the vehicle qualifies for clean fuel incentives.
# - **Electric Range:** Estimated miles per full charge.
# - **Base MSRP:** Manufacturer’s Suggested Retail Price.
# - **Legislative District:** Legislative district where the vehicle is registered.
# - **DOL Vehicle ID:** Unique identifier assigned by the Department of Licensing.
# - **Vehicle Location:** Latitude and longitude of the registered vehicle.
# - **Electric Utility:** Electric utility provider for the vehicle’s registered location.
# - **2020 Census Tract:** Census tract based on 2020 census data.
# 

# ### 📥 Importing Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 📥 Loading the Dataset

# In[2]:


df=pd.read_csv('Electric_Vehicle_Population_Data.csv')
df.head()


# ### 🔍 Displaying basic information about the dataset
# 

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# ## 4️⃣ Data Preprocessing & Cleaning

# ### 🛠️ Checking for Missing Values

# In[6]:


df.isnull().sum()


# ### 🛠️ Checking for Dublicate Values

# In[7]:


df.duplicated().sum()


# ### 🔧 Data Cleaning

# In[8]:


# Handling missing values
df=df.dropna()


# In[9]:


# Standardizing column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# In[10]:


# Standardizing categorical text values
df['make'] = df['make'].str.title()
df['model'] = df['model'].str.upper()


# In[11]:


df.sample(5)


# ## 5️⃣ Exploratory Data Analysis (EDA)

# ### 📆 EV Adoption Over Time

# In[24]:


plt.figure(figsize=(10, 5))
ev_adoption = df['model_year'].value_counts().sort_index()
sns.lineplot(x=ev_adoption.index, y=ev_adoption.values, marker='o', color='b')
plt.title("EV Adoption Over Time")
plt.xlabel("Model Year")
plt.ylabel("Number of EVs Registered")
plt.grid(True)

plt.show()


# ### 🗺️ Geographical Distribution of EVs

# In[25]:


plt.figure(figsize=(12,6))
top_cities = df['city'].value_counts().head(10)
sns.barplot(x=top_cities.values, y=top_cities.index, palette='coolwarm')
plt.title("Top 10 Cities with the Most EV Registrations")
plt.xlabel("Number of EVs")
plt.ylabel("City")

plt.show()


# ### ⚡ Electric Range Analysis

# In[26]:


plt.figure(figsize=(10,5))
sns.histplot(df['electric_range'], bins=20, kde=True, color='green')
plt.title("Distribution of Electric Vehicle Ranges")
plt.xlabel("Electric Range (Miles)")
plt.ylabel("Frequency")

plt.show()


# ### 🔋 Distribution of EV Types with Percentage

# In[27]:


plt.figure(figsize=(8,5))
ax = sns.countplot(x=df['electric_vehicle_type'], palette='coolwarm')

# Adding percentages on bars
total = len(df)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 50), ha='center', fontsize=12, color='black')

plt.title("Distribution of EV Types")
plt.xlabel("EV Type")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# ### 🚘 Top 10 EV Manufacturers

# In[28]:


plt.figure(figsize=(12,6))
top_makers = df['make'].value_counts().nlargest(10)  # Get top 10 manufacturers

ax = sns.barplot(x=top_makers.index, y=top_makers.values, palette='viridis')

# Adding value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 500), ha='center', fontsize=12, color='black')

plt.title("Top 10 EV Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Number of Registered EVs")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# ### ⚡ Distribution of Electric Vehicle Ranges

# In[29]:


plt.figure(figsize=(12,6))
sns.histplot(df['electric_range'], bins=30, kde=True, color='blue')

plt.title("Distribution of Electric Vehicle Ranges")
plt.xlabel("Electric Range (Miles)")
plt.ylabel("Number of Vehicles")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# ### 📈 Estimated Growth in Market Size

# #### 📊 Aggregate EV count by model year

# In[18]:


ev_trend = df['model_year'].value_counts().sort_index()  # Count EV registrations per year
years = ev_trend.index.values  # Extract years
ev_count = ev_trend.values  # Extract EV counts


# #### 🔄 Convert data to NumPy arrays for mathematical operations

# In[19]:


years = np.array(years, dtype=int)
ev_count = np.array(ev_count, dtype=int)


# #### 📈 Fit Linear Regression (Degree 1)

# In[20]:


linear_coeffs = np.polyfit(years, ev_count, 1)  # Find best-fit line coefficients
linear_model = np.poly1d(linear_coeffs)  # Create linear model function


# #### 🏎️ Fit Polynomial Regression (Degree 3)

# In[21]:


degree = 3  # Higher degree captures trends better
poly_coeffs = np.polyfit(years, ev_count, degree)  # Find best-fit polynomial coefficients
poly_model = np.poly1d(poly_coeffs)  # Create polynomial model function


# #### 📅 Generate Predictions for Future Years (2026-2030)

# In[22]:


future_years = np.arange(2026, 2031)  # Define future years for prediction
linear_predictions = linear_model(future_years)  # Predict future EV counts (linear model)
poly_predictions = poly_model(future_years)  # Predict future EV counts (polynomial model)


# #### 🔍 Visualization: Compare Actual Data & Predictions

# In[30]:


plt.figure(figsize=(10, 5))  # Set figure size

# 🔵 Scatter plot for actual EV data
sns.scatterplot(x=years, y=ev_count, label="Actual Data", color="blue")

# 🟠 Line plot for Linear Regression
sns.lineplot(x=years, y=linear_model(years), label="Linear Regression", color="orange")

# 🟢 Line plot for Polynomial Regression (Degree 3)
sns.lineplot(x=years, y=poly_model(years), label="Polynomial Regression (Degree 3)", color="green")

# 🔴 Future Predictions using Linear Model
sns.scatterplot(x=future_years, y=linear_predictions, label="Linear Prediction (2026-2030)", color="red", marker="o")

# 🟣 Future Predictions using Polynomial Model
sns.scatterplot(x=future_years, y=poly_predictions, label="Polynomial Prediction (2026-2030)", color="purple", marker="o")

# 📌 Mark Prediction Start (2025) with a vertical dashed line
plt.axvline(x=2025, linestyle="--", color="gray", label="Prediction Start")

# 🏷️ Labels & Title
plt.xlabel("Year")
plt.ylabel("Number of EVs Registered")
plt.title("EV Growth Prediction: Linear vs. Polynomial Regression")
plt.legend()  # Show legend
plt.grid()  # Enable grid


plt.show()


# #### **Techniques Used:**
# 1. **Linear Regression (Degree 1)**: Assumes a constant rate of growth in EV adoption.
# 2. **Polynomial Regression (Degree 3)**: Captures more complex, nonlinear growth trends.
# 
# #### **Why Compare Both?**
# - **Linear Regression** provides a simple trend analysis but may not capture changes in adoption rate.
# - **Polynomial Regression** is more flexible and better models accelerated EV adoption due to technological advancements and government policies.

# #### 📌 Explanation of the Methods Used
# 
# ##### 🔹 **Linear Regression**
# This method assumes a straight-line relationship between the years and EV count.  
# It fits the equation:  
# \[
# EV\_Count = m \times Year + b
# \]
# - Simple to interpret  
# - Works well for trends that increase consistently over time  
# 
# ##### 🔹 **Polynomial Regression (Degree 3)**
# This method fits a curved line to better capture fluctuations in growth.  
# It fits the equation:  
# \[
# EV\_Count = a \times Year^3 + b \times Year^2 + c \times Year + d
# \]
# - Can model complex growth patterns  
# - More flexible but can lead to overfitting  
# 
# ##### 🔹 **Comparison**
# - If EV growth follows a steady increase, **Linear Regression** might be sufficient.  
# - If growth shows fluctuations or acceleration over time, **Polynomial Regression** is a better fit.  
# 
# ✅ The results indicate that while **linear regression provides a basic trend**, the **polynomial model captures variations in growth more effectively**. However, predictions should be interpreted carefully to avoid overfitting.
# 

# ## 6️⃣📌 Conclusion
# 
# ### 🔹 Electric Vehicle Population Analysis
# 
# #### 1️⃣ Summary of Findings
# 
# - **📈 EV Growth Over Time**: The number of registered EVs has increased significantly, indicating a rising adoption trend.
# - **🗺️ Geographical Distribution**: Certain cities and states have a higher concentration of EVs, influenced by policies and charging infrastructure.
# - **🔋 EV Type Distribution**: Battery Electric Vehicles (BEVs) are more common than Plug-in Hybrid Electric Vehicles (PHEVs), showing a shift toward full electrification.
# - **🚘 Popular Manufacturers**: Tesla, Nissan, and other major brands dominate the EV market.
# - **⚡ Electric Range Improvements**: Advancements in battery technology are leading to longer-range EVs, enhancing usability.
# - **📊 Market Growth Prediction**: Polynomial regression suggests an accelerating adoption rate, while linear regression provides a steady growth estimate.
# 
# #### 2️⃣ Key Takeaways
# 
# - The **EV industry is growing rapidly**, with increased adoption driven by environmental concerns, government incentives, and technological improvements.
# - **Charging infrastructure and affordability** remain critical challenges for wider adoption.
# - **Future market expansion** is expected, with further improvements in battery range and cost efficiency.
# 
# #### 3️⃣ Final Thoughts
# 
# With continued advancements in EV technology and supportive government policies, the transition to electric mobility is becoming more feasible. However, addressing infrastructure gaps and affordability challenges will be crucial to sustaining this growth.
# 
