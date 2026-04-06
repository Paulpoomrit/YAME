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

data_path = 'data/test_data/w_features.tsv'
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

def train_model(df: pd.DataFrame, ablated_feature: str):

    feature_list = feature_names

    if ablated_feature in feature_list:
        print(f'removing: {ablated_feature}')
        feature_list.remove(ablated_feature)

    y = df['label']
    X = df[feature_list].values


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=104,
                                                        shuffle=True)


    model = RandomForestClassifier(
        n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
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

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # roc curve for tpr = fpr
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"results/{feature}/yame_wout_{feature}_auroc.png")
    # # plt.show()

    # Plot the predicted class probabilities
    plt.hist(y_pred_prob, bins=10)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of text being labelled "AI"')
    plt.ylabel('Frequency')
    plt.savefig(f"results/{feature}/yame_wout_{feature}_pred_prob.png")
    # # plt.show()

    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"AUROC:  {roc_auc:.2f}")

    print("\n---Feature importance---\n")
    for score, name in zip(model.feature_importances_, df.columns):
        print(f"| {name} | {round(score,2)} |")
    print("\n------------------------\n")

    # Visualize tree
    dot_file_path = f"results/{feature}/yame_wout_{feature}.dot"
    export_graphviz(
        model,
        out_file= dot_file_path,
        feature_names=feature_list,
        class_names={'human', 'AI'},
        filled=True,
        rounded=True,
        special_characters=True
    )



df = pd.read_csv(data_path, sep='\t', index_col=False)
df = df.dropna()

print("\n----------------------------------------\n")
print(f"All features")
train_model(df, 'none')
print("\n----------------------------------------\n")


for feature in feature_names:
    print("\n----------------------------------------\n")
    print(f"Ablated feature: {feature}")
    ablated_df = df.copy()
    ablated_df = ablated_df.drop(feature, axis=1)
    train_model(ablated_df, feature)
    print("\n----------------------------------------\n")

