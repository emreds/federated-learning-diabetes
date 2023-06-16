from sklearn.metrics import confusion_matrix, make_scorer


# Setup classifier scorers
def get_scores(y_true, y_pred):
    conf_results = confusion_matrix(y_true, y_pred)
    scores = {
        "Accuracy": "accuracy",
        "Sensitivity": "recall",
        "Precision": "precision",
        "tp": conf_results[1, 1],
        "tn": conf_results[0, 0],
        "fp": conf_results[0, 1],
        "fn": conf_results[1, 0],
    }

    scores["Sensitivity"] = (
        round(scores["tp"].mean() / (scores["tp"].mean() + scores["fn"].mean()), 3)
        * 100
    )  # TP/(TP+FN) also recall
    scores["Specificity"] = (
        round(scores["tn"].mean() / (scores["tn"].mean() + scores["fp"].mean()), 3)
        * 100
    )  # TN/(TN+FP)
    scores["PPV"] = (
        round(scores["tp"].mean() / (scores["tp"].mean() + scores["fp"].mean()), 3)
        * 100
    )  # PPV = tp/(tp+fp) also precision
    scores["NPV"] = (
        round(scores["tn"].mean() / (scores["fn"].mean() + scores["tn"].mean()), 3)
        * 100
    )  # TN(FN+TN)
    scores["Precision"] = (
        round(scores["tp"].mean() / (scores["tp"].mean() + scores["fp"].mean()), 3)
        * 100
    )  # TP/(TP+FP)
    scores["F1"] = round(
        2
        * (
            (scores["Precision"] * scores["Sensitivity"])
            / (scores["Precision"] + scores["Sensitivity"])
        ),
        3,
    )  # 2*((precision*sensitivity)/(precision+sensitivity))
    return scores
