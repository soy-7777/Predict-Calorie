def self_interaction(df, features):
    drop_features = []
    for name in features:
        drop_features += [f"{name}_x_{name}_sqrt",
                          f"{name}_x_{name}_squared",
                          f"{name}_sqrt_x_{name}_squared"
                         ]
    drop_features = [col for col in drop_features if col in df.columns]
    df.drop(drop_features, axis=1, inplace=True)
    return df
  """
  df.columns：df に存在するすべての列名（pandas の Index 型）。
  if col in df.columns：その列が本当に df にあるかどうか確認。
  [col for col in drop_features if ...]：条件を満たす列だけを取り出して新しいリストにする。
  """

train = self_interaction(train, numerical_features)
test = self_interaction(test, numerical_features)
