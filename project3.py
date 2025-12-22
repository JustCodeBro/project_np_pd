import numpy as np
import pandas as pd


def main():
    """
    Project 3: Simple Credit Scoring System
    Goal: Perform Data Normalization and Matrix Weights to assess loan risk.
    """
    # --- STEP 1: DATA INITIALIZATION ---
    rng = np.random.default_rng(42)

    # Simulating data for 1000 customers
    # Features: [Income (k$), Current Debt (k$), Age, Late Payments (count)]
    # We use different scales to make normalization necessary.
    income = rng.integers(20, 200, size=1000)  # 20k to 200k
    debt = rng.integers(0, 100, size=1000)  # 0k to 100k
    age = rng.integers(18, 70, size=1000)  # 18 to 70 years old
    late_pays = rng.integers(0, 10, size=1000)  # 0 to 10 times

    # Combine into a NumPy array (Matrix)
    raw_data = np.column_stack((income, debt, age, late_pays))

    # Create DataFrame
    cols = ["Income", "Debt", "Age", "Late_Payments"]
    df_customers = pd.DataFrame(raw_data, columns=cols)
    df_customers.index = [f"ID_{i+1}" for i in range(1000)]

    # --- STEP 2: MIN-MAX NORMALIZATION ---
    # Why? Income is ~200 while Late_Payments is ~10.
    # Without normalization, Income would dominate the score unfairly.
    # Formula: (X - min) / (max - min)
    data_min = raw_data.min(axis=0)
    data_max = raw_data.max(axis=0)

    # normalized_data will have all values between 0 and 1
    normalized_data = (raw_data - data_min) / (data_max - data_min)

    # --- STEP 3: CALCULATE CREDIT SCORE (WEIGHTED SUM) ---
    # Weights: Income(+40%), Debt(-30%), Age(+10%), Late_Payments(-20%)
    # Note: Debt and Late_Payments are negative because they INCREASE risk.
    weights = np.array([0.4, -0.3, 0.1, -0.2])

    # Use Dot Product to multiply normalized features by weights
    # Result: A score representing creditworthiness
    score_array = np.dot(normalized_data, weights)

    # Scale score to a 0-100 range for readability
    df_customers["Credit_Score"] = (
        (score_array - score_array.min())
        / (score_array.max() - score_array.min())
        * 100
    ).round(2)

    # --- STEP 4: DECISION MAKING & ANALYSIS ---
    # If score > 50: Approved, else Rejected
    df_customers["Decision"] = np.where(
        df_customers["Credit_Score"] > 50, "Approved", "Rejected"
    )

    # Grouping by Age to see approval rates
    # pd.cut: Segments data into age bins
    df_customers["Age_Group"] = pd.cut(
        df_customers["Age"], bins=[18, 30, 45, 70], labels=["Young", "Middle", "Senior"]
    )

    # Pivot Table: Calculate Approval Rate per Age Group
    pivot = df_customers.pivot_table(
        values="Decision",
        index="Age_Group",
        aggfunc=lambda x: (x == "Approved").mean() * 100,
    )

    # --- STEP 5: FINDING ANOMALIES ---
    # Find customers with High Income (>150k) but were Rejected
    high_earner_rejected = df_customers[
        (df_customers["Income"] > 150) & (df_customers["Decision"] == "Rejected")
    ]

    # --- OUTPUT ---
    print("--- First 5 Customers ---")
    print(df_customers.head())

    print("\n--- Approval Rate (%) by Age Group ---")
    print(pivot)

    print(f"\nNumber of High-Earners Rejected: {len(high_earner_rejected)}")
    if not high_earner_rejected.empty:
        print(
            "Example of a High-Earner Rejected (Likely due to high debt or late payments):"
        )
        print(high_earner_rejected.iloc[0])


if __name__ == "__main__":
    main()
