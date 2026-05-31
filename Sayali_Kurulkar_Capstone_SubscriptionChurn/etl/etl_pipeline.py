import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

DATA_PATH = "./Datasets/"

#Load Datasets
users = pd.read_csv(f"{DATA_PATH}/users.csv")
plans = pd.read_csv(f"{DATA_PATH}/plans.csv")
subscriptions = pd.read_csv(f"{DATA_PATH}/subscriptions.csv")   
payments = pd.read_csv(f"{DATA_PATH}/payments.csv") 
usage_daily = pd.read_csv(f"{DATA_PATH}/usage_daily.csv")
campaigns = pd.read_csv(f"{DATA_PATH}/campaign_touchpoints.csv")  

#Cleaning & Standardization
def normalize_text(series):
    return (
        series
        .astype(str)
        .str.lower()
        .str.strip()
    )

def flag_outliners(series, p99=0.99):
    threshold = series.quantile(p99)
    return series > threshold

#Users Dataset
users = users.drop_duplicates(subset="user_id")

users["signup_date"] = pd.to_datetime(users["signup_date"])

users["city_tier"] = (
    users["city_tier"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
)

users["segment"] = normalize_text(users["segment"])
users["preferred_device"] = normalize_text(users["preferred_device"])

#Plans Dataset
plans["plan_name"] = normalize_text(plans["plan_name"])
plans["billing_cycle_days"] = normalize_text(plans["billing_cycle_days"])

#Subscriptions Dataset
subscriptions = subscriptions.drop_duplicates(
    subset=["user_id", "subscription_id"]
)

subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"])
subscriptions["status"] = normalize_text(subscriptions["status"])

#Payments Dataset
payments = payments.drop_duplicates(
    subset=["payment_id"]
)

payments["payment_date"] = pd.to_datetime(payments["payment_date"])
payments["payment_status"] = normalize_text(payments["payment_status"])

payments["is_failed"] = payments["payment_status"].isin(
    ["failed", "declined", "error"]
).astype(int)

#Usage Daily Dataset
usage_daily["date"] = pd.to_datetime(usage_daily["date"])

usage_daily["minutes_used"] = usage_daily["minutes_used"].fillna(0)
usage_daily["sessions_count"] = usage_daily["sessions_count"].fillna(0)

usage_daily["usaage_outlier_flag"] = flag_outliners(
    usage_daily["minutes_used"]
)

#Output 1:
#Last active date for each user based on usage_daily dataset
last_active = (
    usage_daily[usage_daily["minutes_used"] > 0]
    .groupby("user_id")["date"]
    .max()  
    .reset_index(name="last_active_date")
)

# Lifetime paid motnhs for each user based on payments dataset
paid_months = (
    payments[payments["payment_status"] == "success"]
    .groupby("user_id")
    .size()
    .reset_index(name="lifetime_paid_months")
)

#Engagement bands based on minutes used
usage_daily_agg = (
    usage_daily.groupby("user_id")["minutes_used"]
    .sum()
    .reset_index()
)

usage_daily_agg["engagement_band"] = pd.qcut(
    usage_daily_agg["minutes_used"],
    q=[0, 0.33, 0.66, 1.0],
    labels=["low", "medium", "high"]
)

#build dimensions and fact tables
#Dim Users
dim_users = (
    users
    .merge(last_active, on="user_id", how="left")
    .merge(paid_months, on="user_id", how="left")
    .merge(usage_daily_agg[["user_id", "engagement_band"]], on="user_id", how="left")
)

as_of_date = usage_daily["date"].max() 

dim_users["tenure_days"] = (
    as_of_date - dim_users["signup_date"]
).dt.days

dim_users["lifetime_paid_months"] = dim_users["lifetime_paid_months"].fillna(0).astype(int)

dim_users.to_csv("./data/dim_users_enriched.csv", index=False)

#Output 2:
#Weekly usage aggregation for each user

usage_daily["week_start"] = usage_daily["date"] - pd.to_timedelta(
    usage_daily["date"].dt.weekday, unit="D"
)

weekly_usage_daily = (
    usage_daily.groupby(["user_id", "week_start"])
    .agg(
        active_days_week=("date", "nunique"),
        total_minutes_week=("minutes_used", "sum"),
        total_sessions_week=("sessions_count", "sum"),
        feature_usage_count_week=("feature_events", "count")
    )
    .reset_index()
)

#weekly payments
payments["week_start"] = payments["payment_date"] - pd.to_timedelta(
    payments["payment_date"].dt.weekday, unit="D"
)

weekly_payments = (
    payments.groupby(["user_id", "week_start"])
    .agg(
        total_attempts_week=("payment_id", "count"),
        payment_failures_week=("is_failed", "sum")
    )
    .reset_index()
)

#Renewal Due Flag
subscriptions["renewal_due_date"] = subscriptions["end_date"] - timedelta(days=7)

renewals = subscriptions[["user_id", "renewal_due_date"]]

#Final weekly fact table

fact_user_weekly = (
    weekly_usage_daily
    .merge(weekly_payments, on=["user_id", "week_start"], how="left")
) 

fact_user_weekly = fact_user_weekly.merge(
    renewals,
    left_on="user_id",
    right_on="user_id",
    how="left"
)

fact_user_weekly["renewal_due_flag"] = (
    (fact_user_weekly["week_start"] >= fact_user_weekly["renewal_due_date"]) 
).astype(int)

fact_user_weekly.fillna(0, inplace=True)

fact_user_weekly.to_csv("./data/fact_user_weekly.csv", index=False)

#------------------
#Output 3:
#------------------

#churn if not renewed within 7 days after expiry
subscriptions["churn_date"] = subscriptions["end_date"] + timedelta(days=7)

churn_labels = (
    subscriptions.groupby("user_id")["churn_date"]
    .min()
    .reset_index()
)

#Feature engineering for churn prediction

LOOKBACK_WEEKS = 4

latest_week = fact_user_weekly["week_start"].max()
cutoff = latest_week - timedelta(weeks=LOOKBACK_WEEKS)

features = fact_user_weekly[
    fact_user_weekly["week_start"] >= cutoff
]

agg_features = (
    features.groupby("user_id")
    .agg(
        avg_minutes_4w=("total_minutes_week", "mean"),
        last_week_minutes=("total_minutes_week", "last"),
        payment_failures_4w=("payment_failures_week", "sum"),
        active_days_4w=("active_days_week", "mean")
    )
    .reset_index()
)

agg_features["usage_trend"] = (
    agg_features["last_week_minutes"] /
    agg_features["avg_minutes_4w"].replace(0, np.nan)
)

#Label creation
as_of_date = latest_week

churn_labels["will_churn_14d"] = (
    (churn_labels["churn_date"] > as_of_date) &
    (churn_labels["churn_date"] <= as_of_date + timedelta(days=14))
).astype(int)

#Final model Dataset

model_dataset = (
    agg_features
    .merge(dim_users, on="user_id", how="left")
    .merge(churn_labels[["user_id", "will_churn_14d"]], on="user_id", how="left")
)

model_dataset["days_since_last_activity"] = (
    as_of_date - model_dataset["last_active_date"]
).dt.days

model_dataset.to_csv(
    "./data/model_churn_dataset.csv",
    index=False
)
