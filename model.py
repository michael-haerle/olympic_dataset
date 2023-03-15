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