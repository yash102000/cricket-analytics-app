import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def run_eda():
    st.title("🔮 Smart IPL Match Outcome Predictor")
    st.subheader("Powered by Machine Learning & Past IPL Data")


    # Load match data
    matches = pd.read_csv("data/matches.csv")

    # Show column names
    st.markdown("📌 **Columns in `matches.csv`:**")
    st.dataframe(pd.DataFrame(matches.columns, columns=["Column Names"]))

    # Optional: Column descriptions (clean UI)
    column_info = {
        "id": "Match ID",
        "Season": "IPL Season Year",
        "city": "City where match was played",
        "date": "Match date",
        "team1": "Team 1",
        "team2": "Team 2",
        "toss_winner": "Team that won the toss",
        "toss_decision": "Decision taken after toss (bat/field)",
        "result": "Match result (normal/tie/no result)",
        "dl_applied": "Duckworth–Lewis rule applied (1 = yes)",
        "winner": "Winning team",
        "win_by_runs": "Margin of victory (runs)",
        "win_by_wickets": "Margin of victory (wickets)",
        "player_of_match": "Player awarded best performance",
        "venue": "Stadium name",
        "umpire1": "First on-field umpire",
        "umpire2": "Second on-field umpire",
        "umpire3": "Third umpire (if any)"
    }

    with st.expander("📘 Column Descriptions", expanded=False):
        column_df = pd.DataFrame(column_info.items(), columns=["Column Name", "Description"])
        st.dataframe(column_df)

    # Extract year
    if 'date' in matches.columns:
        matches['year'] = pd.to_datetime(matches['date'], dayfirst=True, errors='coerce').dt.year
    else:
        st.error("❌ Could not find a 'date' column in the dataset.")
        return

    season_counts = matches['year'].value_counts().sort_index()

    # Plot using Seaborn
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=season_counts.index.astype(str), y=season_counts.values, ax=ax)

    ax.set_title("🏏 IPL Matches per Season", fontsize=16)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Number of Matches", fontsize=12)

    for i, val in enumerate(season_counts.values):
        ax.text(i, val + 1, str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    st.pyplot(fig)
