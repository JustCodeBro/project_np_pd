import numpy as np
import pandas as pd


def main():
    """
    Project 2: Student Performance Analysis & Scholarship Evaluation
    Goal: Mastering Weighted Averages (NumPy) and Conditional Mapping (Pandas).
    """

    # --- STEP 1: DATA INITIALIZATION ---
    # Using the modern Generator API (recommended over np.random.seed)
    rng = np.random.default_rng(42)

    # Generate random scores. Note: uniform(0, 10) creates values in half-open interval [0, 10)
    scores_numpy = rng.uniform(0, 10, size=(100, 3)).round(1)

    columns = ["Math", "Physics", "Chemistry"]
    df_students = pd.DataFrame(scores_numpy, columns=columns)
    df_students.index = [f"Student_{i + 1}" for i in range(100)]

    # --- STEP 2: CALCULATE WEIGHTED AVERAGE ---
    weights = np.array([2, 1.5, 1])

    # Dot product efficiently handles (Math*2 + Physics*1.5 + Chemistry*1) for all rows
    weighted_sum = np.dot(scores_numpy, weights)
    print(weighted_sum)
    df_students["Avg_Score"] = (weighted_sum / weights.sum()).round(2)

    # --- STEP 3: STUDENT CLASSIFICATION ---
    conditions = [
        (df_students["Avg_Score"] >= 9.0),
        (df_students["Avg_Score"] >= 8.0),
        (df_students["Avg_Score"] >= 6.5),
        (df_students["Avg_Score"] < 6.5),
    ]
    grades = ["Excellent", "Good", "Fair", "Average"]

    # np.select is a vectorized way to apply multiple conditions
    df_students["Grade"] = np.select(conditions, grades, default="Failed")

    # --- STEP 4: STATISTICS & QUERIES ---
    grade_counts = df_students["Grade"].value_counts()

    # IMPROVED: Using .loc for more efficient and idiomatic filtering.
    # We use .loc[row_indexer, column_indexer] to access data in one clean step.
    math_avg_excellent = df_students.loc[
        df_students["Grade"] == "Excellent", "Math"
    ].mean()

    # Retrieve top 5 scholarship candidates
    top_5_scholarships = df_students.nlargest(5, "Avg_Score")

    # --- OUTPUT RESULTS ---
    print("--- First 5 Rows of Data ---")
    print(df_students.head())

    print("\n--- Student Grade Statistics ---")
    print(grade_counts)

    # Handling potential NaN if no "Excellent" students exist
    if pd.isna(math_avg_excellent):
        print(
            "\nAverage Math Score of Excellent students: No students found in this category."
        )
    else:
        print(f"\nAverage Math Score of Excellent students: {math_avg_excellent:.2f}")

    print("\n--- Top 5 Candidates for Scholarship ---")
    print(top_5_scholarships[["Avg_Score", "Grade"]])


if __name__ == "__main__":
    main()
