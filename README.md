🚀 Projects Overview
1. Store Performance Analytics (Easy)
- Goal: Practice basic data cleaning and time-series aggregation.
    - Key Features: 
        * Simulated 12-month revenue data for 3 branches using np.random.  
         * Handling missing values (NaN) using mean imputation (fillna).
         * Monthly and branch-wise performance comparisons using axis=0 and axis=1.
    + Tech Stack: Pandas (DataFrames, Fillna), NumPy (Randomization).
2. Student Grading & Scholarship System (Medium)
    - Goal: Implement weighted calculations and advanced conditional logic.
        + Key Features: 
            * Weighted Average: Calculated using the Dot Product (np.dot) to account for different subject coefficients (Math, Physics, Chemistry).
            * Multi-condition Mapping: Used np.select to categorize students into "Excellent", "Good", "Fair", or "Average" without nested loops.
            * Top-K Querying: Identified top candidates for scholarships using .nlargest().
    + Tech Stack: NumPy (Linear Algebra/Dot Product, np.select), Pandas (Filtering, value_counts).
3. Credit Scoring & Risk Assessment (Hard)
    - Goal: Simulate a real-world financial decision-making engine using data normalization.
        + Key Features: 
            * Feature Scaling: Implemented Min-Max Normalization to bring Income ($20k-$200k) and Late Payments (0-10) to a common scale [0, 1] 
            * Risk Modeling: Applied custom weights to features (Income, Debt, Age, History) to generate a Credit Score.
            * Segment Analysis: Used pd.cut for binning ages and pivot_table to calculate approval rates per demographic.
            * Anomaly Detection: Found "high-income" individuals rejected due to poor debt-to-income ratios.
        + Tech Stack: NumPy (Feature scaling, Matrix stacking), Pandas (Pivot tables, binning/cutting).🛠
Core Skills Demonstrated
Skill                       Description
Vectorization  |  Avoiding for loops by using NumPy's array-based math for high performance.
Data Cleaning |       Detecting and replacing NaN values to ensure dataset integrity.Feature Engineering   | Creating new metrics (Avg_Score, Credit_Score) from raw data.
AggregationUsing  |     GroupBy and Pivot Tables to extract business insights from large datasets.
Normalization      |    Understanding that data must be scaled before mathematical modeling.

:) How to run file
Note: This folder use uv, so make sure that you install it by using command
pip install uv
pip3 install uv
and then run the file that you want to run using command 
+ uv run project1.py
+ uv run project2.py
+ uv run project3.py