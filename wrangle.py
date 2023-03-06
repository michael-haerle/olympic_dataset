# Standard imports
import pandas as pd

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

# Lets make a function to return a 1 if they received a medal and a 0 if they didn't
def filter(x):
    if x == 'Gold' or x == 'Silver' or x =='Bronze':
        return 1
    else:
        return 0

def filter_2(x):
    if x == 'M':
        return 1
    else:
        return 0


def wrangle_df():
    df = pd.read_csv('athlete_events.csv')
    df1 = pd.read_csv('noc_regions.csv')
    df['Sex_Male'] = df['Sex'].apply(filter_2)
    df['medalist'] = df['Medal'].apply(filter)
    df['BMI'] = round((df['Weight'] * 0.45359237) / ((df['Height'] / 100) ** 2), 1)
    # Lets create bins for the age and add it as a column
    bins= [0,10,20,30,40,50,60,70,80,90]
    labels = ['Under_10','10s','20s','30s','40s','50s','60s','70s','80s']
    df['AgeBins'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.medalist)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=df.medalist)
    imputer = SimpleImputer(strategy='mean')
    imputer = imputer.fit(train[['Age']])
    train[['Age']] = imputer.transform(train[['Age']])

    validate[['Age']] = imputer.transform(validate[['Age']])

    test[['Age']] = imputer.transform(test[['Age']])

    train = train[(train['Year'] > 1960)]
    validate = validate[(validate['Year'] > 1960)]
    test = test[(test['Year'] > 1960)]

    train.Medal = train.Medal.fillna('None')
    validate.Medal = validate.Medal.fillna('None')
    test.Medal = test.Medal.fillna('None')

    train = train.dropna()
    validate = validate.dropna()
    test = test.dropna()

    return df, df1, train, validate, test