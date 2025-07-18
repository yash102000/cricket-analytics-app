# prediction.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run_prediction():
    st.subheader("🎯 Who Will Win? Let's Predict the Winner!")

    matches = pd.read_csv("data/matches.csv")
    df = matches.dropna(subset=['winner'])

    # Prepare encoders
    team_encoder = LabelEncoder()
    toss_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()
    winner_encoder = LabelEncoder()

    # Fit and transform
    df['team1_enc'] = team_encoder.fit_transform(df['team1'])
    df['team2_enc'] = team_encoder.transform(df['team2'])
    df['toss_winner_enc'] = toss_encoder.fit_transform(df['toss_winner'])
    df['venue_enc'] = venue_encoder.fit_transform(df['venue'])
    df['winner_enc'] = winner_encoder.fit_transform(df['winner'])

    # Features and target
    features = ['team1_enc', 'team2_enc', 'toss_winner_enc', 'venue_enc']
    X = df[features]
    y = df['winner_enc']

    # Train-test split for better performance tracking (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Unique dropdown options
    teams = sorted(matches['team1'].dropna().unique())
    venues = sorted(matches['venue'].dropna().unique())

    # User Inputs
    team1 = st.selectbox("🏏 Choose Team 1", teams)
    team2 = st.selectbox("🏏 Choose Team 2", [t for t in teams if t != team1])
    venue = st.selectbox("📍 Match Venue", venues)
    toss_winner = st.radio("🪙 Toss Winner", [team1, team2])

    if st.button("Predict Winner"):
        # Encode user inputs
        input_data = pd.DataFrame({
            'team1_enc': [team_encoder.transform([team1])[0]],
            'team2_enc': [team_encoder.transform([team2])[0]],
            'toss_winner_enc': [toss_encoder.transform([toss_winner])[0]],
            'venue_enc': [venue_encoder.transform([venue])[0]]
        })

        # Prediction & probabilities
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        predicted_team = winner_encoder.inverse_transform([prediction])[0]
        confidence = round(np.max(proba) * 100, 2)

        # Show Results
        st.success(f"🏅 **Predicted Winner: {predicted_team}**")
        st.info(f"🔍 Prediction Confidence: **{confidence}%**")

        # Explanation card
        with st.expander("🧠 Why this prediction?"):
            st.markdown(f"""
                ✅ **Selected Teams:** {team1} vs {team2}  
                ✅ **Toss Winner:** {toss_winner}  
                ✅ **Venue:** {venue}  
                🔎 Based on past IPL data and these conditions, the model predicts **{predicted_team}** as most likely to win this match.
            """)
