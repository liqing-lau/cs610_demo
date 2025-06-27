# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost

from functions import get_number_of_days_stayed, preprocessing

# Load model and scaler
model = joblib.load("pickle_files/xgb_model.pkl")

st.title("Hotel Cancellation Prediction App")

date_fields = ["Date of Arrival", "Date of Depature"]

fields = ["Lead time", "Adults", "Children", "Babies", "Booking changes",
        "Average Daily Rate", 'Number of special request', "Number of previous cancellation", 
        'Number of bookings not cancelled', 'Total number of special requests', 'Number of car parking space']

multiple_choices = ["Hotel Type", 'Deposit Type', "Country", 'Meal Plan',  
                    "Market Segment", 'Distribution Channel', 'Customer Type',
                    "Resevred room type", "Assigned room type"]

boolean_choices = ['Is repeated guest?']

user_input = {}

# Collecting date inputs
arrival_date = st.date_input(date_fields[0])
departure_date = st.date_input(date_fields[1], min_value=arrival_date, value=arrival_date)

# Collecting numerical inputs
lead_time = st.number_input(fields[0], min_value=0, step=1)
adults = st.number_input(fields[1], min_value=1, step=1)
children = st.number_input(fields[2], min_value=0, step=1)
babies = st.number_input(fields[3], min_value=0, step=1)
booking_changes = st.number_input(fields[4], min_value=0, step=1, max_value=3)
adr = st.number_input(fields[5], min_value=0.0, step=0.1, max_value=300.0, value=79.0)
previous_cancellations = st.number_input(fields[7], min_value=0, step=1, max_value=1)
previous_booking_not_cancelled = st.number_input(fields[8], min_value=0, step=1, max_value=5)
# agent = st.number_input(fields[9], min_value=0, step=1)
# company = st.number_input(fields[10], min_value=0, step=1)
# days_in_waiting_list = st.number_input(fields[11], min_value=0, step=1)
total_of_special_requests = st.number_input(fields[9], min_value=0, step=1, max_value=3)
number_of_car_bookings = st.number_input(fields[10], min_value=0, step=1, max_value=1)

# Multiple choices (dropdowns or checkboxes)
hotel_type = st.selectbox(multiple_choices[0], options=['Hotel', 'Resort'])
deposit_type = st.selectbox(multiple_choices[1], options=['No Deposit', 'Non Refund', 'Refundable'])
country = st.selectbox(multiple_choices[2], options=['PRT','GBR','FRA','ESP','DEU',"IRL", 'ITA', "BEL", "NLD", "USA"])
meal_plan = st.selectbox(multiple_choices[3], options=['BB', 'FB', 'HB', 'SC'])
market_segment = st.selectbox(multiple_choices[4], options=['Corporate', 'Direct', 'GDS', 'TA/TO', 'Undefined'])
distribution_channel = st.selectbox(multiple_choices[5], options=['Direct','Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Aviation', 'Undefined'])
customer_type = st.selectbox(multiple_choices[6], options=['Contract', 'Group', 'Transient', 'Transient-Party'], index=2)
reserved_room_type = st.selectbox(multiple_choices[7], options=["A","B","C","D","E","F","G","H","L","P"], index=2)
assigned_room_type = st.selectbox(multiple_choices[8], options=["A","B","C","D","E","F","G","H","L","P"], index=2)

# Boolean choices (checkboxes)
is_repeated_guest = st.checkbox(boolean_choices[0])

weekends, weekdays = get_number_of_days_stayed(arrival_date, departure_date)

# Predict
if st.button("Predict"):
    user_input = {
        'hotel': hotel_type, 
        'lead_time': lead_time,  
        'arrival_date_year': arrival_date.year, 
        'arrival_date_month': arrival_date.month,  
        'arrival_date_week_number': str(arrival_date.isocalendar()[1]), 
        'arrival_date_day_of_month': str(arrival_date.day),  
        'stays_in_weekend_nights': weekends, 
        'stays_in_week_nights': weekdays,  
        'adults': adults,  
        'children': children,  
        'babies': babies,  
        'meal': meal_plan, 
        'country': country,  
        'market_segment': market_segment,  
        'distribution_channel': distribution_channel, 
        'is_repeated_guest': is_repeated_guest,  
        'previous_cancellations': previous_cancellations, 
        'previous_bookings_not_canceled': previous_booking_not_cancelled,  
        'reserved_room_type': reserved_room_type, 
        'assigned_room_type': assigned_room_type,  
        'booking_changes': booking_changes,  
        'deposit_type': deposit_type,  
        'customer_type': customer_type,  
        'adr': adr, 
        'required_car_parking_spaces': number_of_car_bookings,  
        'total_of_special_requests': total_of_special_requests, 
    }

    df = preprocessing(user_input)
    prediction = model.predict(df)
    prediction_probability = model.predict_proba(df)

    if prediction[0]: 
        probability = prediction_probability[0][1] * 100
        st.error(f"Probability of cancellation: {probability:.2f}%")
    else: 
        probability = prediction_probability[0][1] * 100
        st.success(f"Probability of cancellation: {probability:.2f}%")

