"""quantitative.py: experiment with modeling the quantitative metadata."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

categorical_df = pd.read_excel("TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx", index_col=0)
enc = OneHotEncoder(sparse_output=False)
one_hot_arr = enc.fit_transform(categorical_df)

quantitative_df = pd.read_excel("TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx", index_col=0)
sns.heatmap(quantitative_df.corr())
plt.savefig("quantitative_corr.png")
plt.close()

values_df = pd.read_excel("TRAIN/TRAINING_SOLUTIONS.xlsx", index_col=0)

# Turn binary tuples into a 4 category problem.
# (No ADHD, Not F) = 0
# (ADHD, Not F) = 1
# (Not ADHD, F) = 2
# (ADHD, F) = 3
values_df = values_df.join(
    pd.Series(values_df.ADHD_Outcome + (values_df.Sex_F * 2), name="combined_category")
)

X_train, X_test, y_train, y_test = train_test_split(
    quantitative_df, values_df.combined_category
)

clf = HistGradientBoostingClassifier().fit(X_train, y_train)
clf.score(X_test, y_test)

max_acc = 0
max_state = 0
for random_state in tqdm(range(0, 2**8)):
    X_train, X_test, y_train, y_test = train_test_split(
        quantitative_df, values_df.ADHD_Outcome, random_state=random_state
    )
    clf = HistGradientBoostingClassifier().fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    if acc > max_acc:
        max_acc = acc
        max_state = random_state

# clf = HistGradientBoostingClassifier()
# cv_res = cross_validate(
#     clf,
#     quantitative_df,
#     values_df.ADHD_Outcome,
#     scoring=("precision", "recall", "accuracy", "f1"),
#     return_estimator=True,
# )

# test = pd.read_excel("TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx", index_col=0)
# res = clf.predict(test)

param_grid = {
    "learning_rate": [i / 10 for i in range(1, 10)],
    #    "max_iter": range(10, 1000, 10),
    "l2_regularization": [i / 10 for i in range(0, 50)],
}

cv = GridSearchCV(HistGradientBoostingClassifier(), param_grid, verbose=1, n_jobs=8)
cv_res = cv.fit(quantitative_df, values_df.combined_category)
