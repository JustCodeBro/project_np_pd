import numpy as np
import pandas as pd


def main():
    """Creating the project 1"""
    rng = np.random.default_rng(seed=67)
    arr = rng.integers(low=0, high=11, size=(12, 3))
    index = [i for i in range(1, 13)]
    columns = ["HN", "HCM", "DN"]
    df = pd.DataFrame(data=arr, index=index, columns=columns)
    # Making some value NaN(the pro way)
    # Generate a boolean mask with ~15% True values.
    # *df.shape unpacks the dimensions tuple into individual arguments for np.random.rand. like this ex: (10,5) -> 10,5
    mask = np.random.rand(*df.shape) < 0.15
    # np.random.rand() = chosing number from 0 to 1(type: floating)
    # If that value is more than 0.15(15%), than set that to true
    print(mask)
    df = df.mask(mask)
    # .mask() : fill all the values TRUE with the text(number,...) that the user type
    # If the user type is empty(or doesn't have) it automatically set it as NaN(Not a Number)
    print(f"\nThe dataframe when had some value NaN:\n{df}")
    df_filled = df.fillna(df.mean()).round(
        1
    )  # .fillna() : fill all the NaN value with the module df.mean() have give
    # .round(<value>) round the float type so it has just 1 decimal
    print(f"\nThe dataframe has been filled with module df.mean() :\n{df_filled}")

    the_revenue_of_branch = (
        df_filled.sum()
    )  # If the axis is not give, it will be axis=0
    print(
        f"\nThe sum of every single branch in the DataFrame: \n{the_revenue_of_branch}"
    )
    df_filled["Sum_Month"] = df_filled.sum(axis=1)  # axis =1 means ----- -> line
    # axis = 0 ||| -> straight
    month_max = df_filled["Sum_Month"].idxmax()
    value_max = df_filled["Sum_Month"].max()
    print(
        f"The highest revenue of month is month: {month_max} with the value {value_max:.2f} millions."
    )


if __name__ == "__main__":
    main()
