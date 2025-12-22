import numpy as np
import pandas as pd


def main():
    """
    Project 2: Student Performance Analysis & Scholarship Evaluation
    Goal: Mastering Weighted Averages (NumPy) and Conditional Mapping (Pandas).
    """

    # --- STEP 1: DATA INITIALIZATION ---
    # np.random.default_rng(42): Creates a random number generator with a seed.
    # The seed ensures that every time you run the code, you get the same 'random' numbers.
    rng = np.random.default_rng(42)

    # rng.uniform(3, 10, size=(100, 3)): Generates a 100x3 matrix of floats between 3 and 10.
    # .round(1): Rounds each score to 1 decimal place.
    scores_numpy = rng.uniform(3, 10, size=(100, 3)).round(1)

    # pd.DataFrame: Converts the NumPy array into a structured table.
    columns = ["Math", "Physics", "Chemistry"]
    df_students = pd.DataFrame(scores_numpy, columns=columns)

    # List comprehension: Sets custom index labels from SV_1 to SV_100.
    df_students.index = [f"Student_{i+1}" for i in range(100)]

    # --- STEP 2: CALCULATE WEIGHTED AVERAGE ---
    # Weighted system: Math (2), Physics (1.5), Chemistry (1).
    weights = np.array([2, 1.5, 1])

    # np.dot(matrix, vector): Performs a Dot Product.
    # How it works: For each row, it calculates (Math*2 + Physics*1.5 + Chemistry*1).
    # Then we divide by weights.sum() (4.5) to get the final average.
    # This is much faster than using a 'for' loop across 100 rows.
    weighted_sum = np.dot(scores_numpy, weights)
    df_students["Avg_Score"] = (weighted_sum / weights.sum()).round(2)

    # --- STEP 3: STUDENT CLASSIFICATION ---
    # Defining conditions for grading.
    # Order matters here: np.select evaluates from top to bottom.
    conditions = [
        (df_students["Avg_Score"] >= 9.0),
        (df_students["Avg_Score"] >= 8.0),
        (df_students["Avg_Score"] >= 6.5),
        (df_students["Avg_Score"] < 6.5),
    ]
    grades = ["Excellent", "Good", "Fair", "Average"]

    # np.select(conditions, grades): Assigns a grade based on the first condition met.
    # It acts like a nested 'IF' statement but operates on the whole column at once.
    # df_students["Grade"] = np.select(conditions, grades) = This code will get an error
    df_students["Grade"] = np.select(conditions, grades, default="Other")

    # --- STEP 4: STATISTICS & QUERIES ---
    # .value_counts(): Counts the occurrences of each unique value in the 'Grade' column.
    grade_counts = df_students["Grade"].value_counts()

    # Boolean Masking: Filters rows where Grade is 'Excellent', then picks the 'Math' column to find the mean.
    math_avg_excellent = df_students[df_students["Grade"] == "Excellent"]["Math"].mean()

    # .nlargest(5, "Column"): Sorts the data and retrieves the top 5 rows based on "Avg_Score".
    top_5_scholarships = df_students.nlargest(5, "Avg_Score")

    # --- OUTPUT RESULTS ---
    print("--- First 5 Rows of Data ---")
    print(df_students.head())

    print("\n--- Student Grade Statistics ---")
    print(grade_counts)

    # f-string: {value:.2f} formats the number to 2 decimal places.
    print(f"\nAverage Math Score of Excellent students: {math_avg_excellent:.2f}")

    print("\n--- Top 5 Candidates for Scholarship ---")
    print(top_5_scholarships[["Avg_Score", "Grade"]])


if __name__ == "__main__":
    main()
