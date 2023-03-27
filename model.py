# Standard imports
import pandas as pd
from scipy import stats

# Visulation Imports
import matplotlib.pyplot as plt
import seaborn as sns

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Modeling imports
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import sklearn.preprocessing
import statsmodels.api as sm
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# turn off pink boxes
import warnings
warnings.filterwarnings("ignore")

def data_split(train, validate, test):
    X_cols = ['Sex_Male', 'Age', 'BMI']
    X_train = train[X_cols]
    y_train = train.medalist

    X_validate = validate[X_cols]
    y_validate = validate.medalist

    X_test = test[X_cols]
    y_test = test.medalist

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def model_scores(cm):
    '''
    Function to get all model scores necessary for codeup exercises
    Accepts a confusion matrix, and prints a report with the following:
        Accuracy
        True positive rate
        False positive rate
        True negative rate
        False negative rate 
        Precision
        Recall
        f1-score
        positive support
        negative support
    '''
    
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    ALL = TP + FP + FN + TN
    
    print('Model stats:')
   
    # accuracy
    acc = (TP + TN) / ALL
    print('Accuracy: {:.2f}'.format(acc))
    # true positive rate, also recall
    TPR = recall = TP/ (TP + FN)
    print('True Positive Rate: {:.2f}'.format(TPR))
    # false positive rate
    FPR = FP / (FP + TN)
    print('False Positive Rate: {:.2f}'.format(FPR))
    # true negative rate
    TNR = TN / (TN + FP)
    print('True Negative Rate: {:.2f}'.format(TNR))
    # false negative rate
    FNR = FN / (FN + TP)
    print('Flase Negative Rate: {:.2f}'.format(FNR))
    # precision
    precision = TP / (TP + FP)
    print('Precision: {:.2f}'.format(precision))
    # recall
    print('Recall: {:.2f}'.format(recall))
    # f1
    f1_score = 2 * (precision*recall) / (precision+recall)
    print('f1 score: {:.2f}'.format(f1_score))
    # support
    support_pos = TP + FN
    print('Positive support:',support_pos)
    support_neg = FP + TN
    print('Negative support:',support_neg)
    print('-----------------------------------------')

def decision_tree_train(X_train, y_train):
    tree = DecisionTreeClassifier(max_depth=7, max_features=2, min_samples_leaf=4, criterion="entropy")
    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    cm = confusion_matrix(y_train, y_pred_train)
    model_scores(cm)

    return dt_top_2

def knn_train_top_2(X_train, y_train):
    knn = KNeighborsClassifier()
    params = {'n_neighbors': range(1,7),
            'weights': ['uniform', 'distance'],
            'leaf_size': range(1,20)}

    grid = GridSearchCV(knn, params, cv=5)

    grid.fit(X_train, y_train)

    results = grid.cv_results_
    test_scores = results['mean_test_score']
    params = results['params']
    for p, s in zip(params, test_scores):
        p['score'] = s

    knn_top_2 = pd.DataFrame(params).sort_values(by='score', ascending=False).head(2)

    return knn_top_2

def baseline_acc(train):
    train['baseline'] = 0
    print('Baseline Accuracy', round(accuracy_score(train.medalist, train.baseline), 2)* 100,'%')