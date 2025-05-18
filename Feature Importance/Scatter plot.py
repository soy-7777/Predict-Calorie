import seaborn as sns

for i in range(20):
    top_imp = importance_df["mean"].iloc[i]
    top_feat = features[i]

    plt.figure()
    sns.scatterplot(x=X[top_feat], y=y)
    plt.title(f"{top_feat} vs Calorie")
    plt.xlabel(top_feat)
    plt.ylabel("Colorie")
    plt.show()
