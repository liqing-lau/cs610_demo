from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def get_number_of_days_stayed(arrival_date, departure_date): 
    if isinstance(arrival_date, str):
        arrival_date = datetime.strptime(arrival_date, '%Y-%m-%d')
    if isinstance(departure_date, str):
        departure_date = datetime.strptime(departure_date, '%Y-%m-%d')
    
    # Initialize counters for weekdays and weekends
    weekdays = 0
    weekends = 0
    
    # Loop through each day in the range
    current_date = arrival_date
    while current_date < departure_date:
        # Check if the current day is a weekend or a weekday
        if current_date.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
            weekends += 1
        else:
            weekdays += 1
        current_date += timedelta(days=1)
    
    return weekends, weekdays

def compare_rooms(row):
    if row['reserved_room_type'] == row['assigned_room_type']:
        return 1 # the guest a room and assigned to the same room
    elif row['reserved_room_type'] != row['assigned_room_type']:
        return 0 # downgraded or upgraded

def preprocessing(dic):
    df = pd.DataFrame([dic])

    df['log_lead_time'] = np.log1p(df['lead_time'])
    df['room_type_same'] = df.apply(compare_rooms, axis=1)
    df['length_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['is_weekend'] = df['stays_in_weekend_nights'].apply(lambda x: 1 if x > 0 else 0)
    df['is_high_season'] = df['arrival_date_month'].isin(['June', 'July', 'August']).astype(int)
    df['season'] = df['arrival_date_month'].map({
                                                    'January': 'Winter', 'February': 'Winter', 'March': 'Spring',
                                                    'April': 'Spring', 'May': 'Spring', 'June': 'Summer',
                                                    'July': 'Summer', 'August': 'Summer', 'September': 'Fall',
                                                    'October': 'Fall', 'November': 'Fall', 'December': 'Winter'
                                                })
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['meal_cost_level'] = df['meal'].map({'SC': 0, 'BB': 1, 'HB': 2, 'FB': 3})
    df['has_meal_plan'] = (df['meal'] != 'SC').astype(int)
    df['meal_deposit_interaction'] = df['meal'].map({'SC': 0, 'BB': 1, 'HB': 2, 'FB': 3}) \
                                        * df['deposit_type'].map({'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2})
    df['deposit_lead_interaction'] = df['deposit_type'].map({'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2}) * df['lead_time']
    df['cancel_history_ratio'] = df['previous_cancellations'] / (df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1)  
    df['cancellation_lead_interaction'] = df['cancel_history_ratio'] * df['lead_time']
    df = df.drop(columns = ['previous_cancellations', 'previous_bookings_not_canceled', 'lead_time'])
    df = df.drop(columns = ['arrival_date_year', 'arrival_date_month'])

    jse = joblib.load("pickle_files/jse.pkl")
    df['country_encoded'] = jse.transform(df[['country']])
    df = df.drop(columns=['country']).reset_index(drop=True)

    categorical_cols = ['hotel',
                        'arrival_date_week_number',
                        'arrival_date_day_of_month',
                        'meal',
                        'market_segment',
                        'distribution_channel',
                        'reserved_room_type',
                        'assigned_room_type',
                        'deposit_type',
                        'customer_type',
                        'season']
    
    encoder = joblib.load("pickle_files/onehot.pkl")
    df_encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df = pd.concat([df.drop(categorical_cols, axis=1), df_encoded], axis=1)

    scaler = joblib.load("pickle_files/scaler.pkl")
    df = scaler.transform(df)
    return df