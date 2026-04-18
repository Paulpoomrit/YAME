import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from graphviz import Source
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import classification_report
import pickle
import csv


data_path = 'data/now_data_subset/now_data_subset_w_features.tsv'
feature_names = ["emdash",
                 "emoji",
                 "bold",
                 "title_case",
                 "list",
                 "curly_quotation",
                 "table",
                 "template",
                 "grammar",
                 "vocab",
                 "neg_parallelism",
                 "rule_of_three",
                 "basic_copulative",
                 "eleg_variation",
                 "subject_line",
                 "summary"
                 ]

_result_fields = ['ablated_feature','accuracy', 'precision', 'recall', 'f1', 'auroc']
_results_dict: list[dict[str, str]] = []

_regression_fields = ['model', 'true_y', 'pred_y', 'correct']
_regression_dict: list[dict[str, str]] = []


def train_model(df: pd.DataFrame, ablated_feature: str):

    result = {}
    feature_list = feature_names.copy()

    if ablated_feature in feature_list:
        print(f'removing: {ablated_feature}')
        feature_list.remove(ablated_feature)

    y = df['label']
    X = df[feature_list].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=104,
                                                        shuffle=True)

    # test = np.random.rand(*X_train.shape)
    # X_train = test

    model = RandomForestClassifier(
        n_estimators=40,
        max_features="log2",
        max_leaf_nodes=5,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        min_samples_split=10
        # criterion='entropy'
        )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # print(y_pred)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="AI")
    recall = recall_score(y_test, y_pred, pos_label="AI")
    f1 = f1_score(y_test, y_pred, pos_label="AI")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label="AI")
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    result['ablated_feature'] = ablated_feature
    result['accuracy'] = accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    result['auroc'] = roc_auc
    _results_dict.append(result)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # roc curve for tpr = fpr
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{ablated_feature}/yame_wout_{ablated_feature}_auroc.png")
    # # plt.show()
    plt.clf()

    # Plot the predicted class probabilities
    plt.hist(y_pred_prob, bins=10)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of text being labelled "AI"')
    plt.ylabel('Frequency')
    plt.savefig(f"results/{ablated_feature}/yame_wout_{ablated_feature}_pred_prob.png")
    # # plt.show()
    plt.clf()

    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"AUROC:  {roc_auc:.2f}")

    print(classification_report(y_pred, y_test))

    print("\n---Feature importance---\n")
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_list, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
    print(feature_imp_df)
    print("\n------------------------\n")

    # Visualize tree
    dot_file_path = f"results/{ablated_feature}/yame_wout_{ablated_feature}.dot"
    export_graphviz(
        model.estimators_[0],
        out_file=dot_file_path,
        feature_names=feature_list,
        class_names= ['human', 'AI'],
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Save full model
    if (ablated_feature=="full"):
        with open ('model/yame.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Record data for regression
    for i, true_y in enumerate(y_test):
        # _regression_fields = ['model', 'true_y', 'pred_y', 'correct']
        data_point = {}
        data_point['model'] = ablated_feature
        data_point['true_y'] = true_y
        data_point['pred_y'] = y_pred[i]
        data_point['correct'] = true_y == y_pred[i]
        _regression_dict.append(data_point)


def main():
    with open('results/results.txt', 'w') as sys.stdout:
        df = pd.read_csv(data_path, sep='\t', index_col=False)
        df = df.dropna()

        print("\n----------------------------------------\n")
        print(f"All features")
        train_model(df, 'full')
        print("\n----------------------------------------\n")

        for feature in feature_names:
            print("\n----------------------------------------\n")
            print(f"Ablated feature: {feature}")
            ablated_df = df.copy()
            ablated_df = ablated_df.drop(feature, axis=1)
            train_model(ablated_df, feature)
            print("\n----------------------------------------\n")

        with open('results/eval.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=_result_fields)
            writer.writeheader()
            writer.writerows(_results_dict)

        with open('results/yame_ablation.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=_regression_fields)
            writer.writeheader()
            writer.writerows(_regression_dict)

main()