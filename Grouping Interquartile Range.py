df["BMI_group"] = pd.qcut(df["BMI"], q=4, labels=["low", "mid_low", "mid_high", "high"])
