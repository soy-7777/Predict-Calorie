FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    model = XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9,
                                n_estimators=2000, learning_rate=0.02, gamma=0.01,
                                max_delta_step=2, early_stopping_rounds=100,
                                eval_metric="rmse", enable_categorical=True, random_state=42)
    results = {"pred": np.zeros(len(X_test)), "rmsle": []}

all_importances = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
   
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
    y_pred_val = model.predict(x_val)
    y_pred_test = model.predict(X_test)
    results["pred"] += y_pred_test / FOLDS
    score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
    results["rmsle"].append(score)
    print(f"Fold {fold + 1} RMSLE: {score:.5f}")

    
    booster = model.get_booster()
    fscore = booster.get_score(importance_type="total_gain")
    fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)

    features, scores = zip(*fscore)
    fscore = dict(zip(features, scores))
    imp_df = pd.DataFrame.from_dict(fscore, orient="index", columns=[f"fold_{fold}"])
    all_importances.append(imp_df)

importance_df = pd.concat(all_importances, axis=1).fillna(0)
importance_df["mean"] = importance_df.mean(axis=1)
importance_df['std'] = importance_df.std(axis=1)
importance_df.sort_values("mean", ascending=False)
print(importance_df)

plt.figure(figsize=(10, len(features)*0.4))
plt.barh(features, scores)
plt.xlabel("Total Gain")
plt.title(f"XGRegressor Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

