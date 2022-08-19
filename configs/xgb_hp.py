from scipy.stats import uniform, randint

def get_hp_space():
    return {
        "colsample_bytree": uniform(0.0, 1.0),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 8),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
        "min_child_weight": uniform(1.0, 10),
    }