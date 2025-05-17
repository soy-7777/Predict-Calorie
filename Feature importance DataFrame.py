FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = {
    "CatBoost": CatBoostRegressor(verbose=0, random_seed=42, cat_features=["Sex"],
                                  early_stopping_rounds=100),
    "XGBoost": XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9,
                            n_estimators=2000, learning_rate=0.02, gamma=0.01,
                            max_delta_step=2, early_stopping_rounds=100,
                            eval_metric="rmse", enable_categorical=True, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10,
                             colsample_bytree=0.7, subsample=0.9, random_state=42, verbose=-1)
}
results = {name: {"pred": np.zeros(len(X_test)), "rmsle": []} for name in models}

for name, model in models.items():
    all_importances = []
    print(f"\n Training {name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if name == "XGBoost":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
            booster = model.get_booster()
            fscore = booster.get_score(importance_type="total_gain")
            fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)

        elif name == "CatBoost":
            model.fit(x_train, y_train, eval_set=(x_val, y_val))
            feature_names = model.feature_names_
            importances = model.get_feature_importance(type="PredictionValuesChange")
            fscore = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
        else:
            model.fit(x_train, y_train)
            importances = model.booster_.feature_importance(importance_type="gain")
            feature_names = model.booster_.feature_name()
            fscore = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            

        y_pred_val = model.predict(x_val)
        y_pred_test = model.predict(X_test)
        results[name]["pred"] += y_pred_test / FOLDS
        score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
        results[name]["rmsle"].append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")
        

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
    plt.title(f"{name} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
