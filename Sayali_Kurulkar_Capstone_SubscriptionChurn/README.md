Subscription Churn Early-Warning & Retention Strategy

Project Overview:
This project analyzes subscriber behaviour to identify the main drivers of churn and proposes a data-driven retention strategy. The solution includes:
    1. Data cleaning and transforming using a Python ETL pipeline.
    2. Creation of curated analytics datasets
    3. Churn diagnosis and KPI analysis
    4. A machine learning churn prediction dataset
    5. A targeted retention strategy and impact estimation
The objective is to enable the comapny to identify users likely to churn 7–14 days in advance and intervene early to improve retention and protect revenue.

Churn Definition & Prediction Horizon

Churn Definition
A user is considered churned if:
    The subscription is cancelled OR the subscription expires and is not renewed within 7 days after expiry.

This definition is applied consistently across analytics and reporting.

Prediction Horizon

The churn prediction dataset is designed to predict:
“Will the user churn in the next 14 days?”

This enables the business to take proactive retention actions before churn occurs.

How to Run the ETL Pipeline

The ETL pipeline processes raw datasets and generates analytics-ready outputs.

Step 1 — Install Dependencies
Ensure Python packages are installed:

"pip install pandas numpy"

Step 2 — Place Raw Data Files
Store all raw datasets inside:
/Datasets/

Required files:
    users.csv
    plans.csv
    subscriptions.csv
    payments.csv
    usage_daily.csv
    campaign_touchpoints.csv

Step 3 — Run the ETL Script
Execute the ETL pipeline:

"python etl_pipeline.py"

The script will:
   1. Clean and standardize raw data
   2. Handle missing values
   3. Remove duplicate records
   4. Generate curated datasets for analytics and modeling.

Curated Outputs Generated

The ETL pipeline produces the following datasets inside:
/data/

1. dim_users_enriched.csv
Granularity: 1 row per user

Contains user-level attributes and derived features.

Key fields:
    user_id
    signup_date
    city_tier
    segment
    preferred_device

Derived features:
    tenure_days
    lifetime_paid_months
    last_active_date
    engagement_band (low / medium / high)

fact_user_weekly.csv
Granularity: 1 row per user per week

Contains aggregated behavioral and payment signals.

Key features:
    active_days_week
    total_minutes_week
    sessions_week
    feature_usage_count_week
    payment_attempts_week
    payment_failures_week
    renewal_due_flag

3. model_churn_dataset.csv
Granularity: 1 row per user

This dataset is prepared for churn prediction modeling.

Key features:
    usage trends (last week vs previous weeks)
    active days
    payment failures
    days since last activity
    plan type and tenure

Label column:
will_churn_14d
Indicates whether the user is expected to churn in the next 14 days.

Dashboard Tool Used

The business dashboard was built using Microsoft Power BI.
The dashboard provides interactive analysis of:

    Churn rate trends
    Retention and cohort analysis
    Usage and engagement patterns
    Churn by segments (plan type, city tier, tenure, engagement band)
    Payment failure trends

Key Deliverables

This project includes:
Python ETL pipeline
Curated analytics datasets
Churn insights and KPI analysis
Churn prediction dataset
Tableau dashboard
Retention strategy with experiment design
30-day business impact estimation

Author
Sayali Kurulkar