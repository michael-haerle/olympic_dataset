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

def correlation_heatmap(train):
    train_corr = train.corr().stack().reset_index(name="correlation")
    g = sns.relplot(
        data=train_corr,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=8, sizes=(100, 250), size_norm=(-.2, .8))
    g.set(xlabel="", ylabel="", title='Olympics Correlation Scatterplot heatmap', aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(45)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

def barplot(train):
    groups = train.groupby('Team').sum()
    groups = groups[groups.medalist > 500].reset_index()
    plt.figure(figsize=(10, 10))
    sns.barplot(x='Team', y='medalist', data=groups, order=groups.sort_values('medalist').Team)
    plt.ylabel('Number of Medals')
    plt.title('Top 7 Teams')

def barplot_medalist_male(train):
    # Increasing the figure size and setting the title
    plt.figure(figsize = (7, 7))
    plt.title('')

    # Plotting the bar plot
    sns.barplot(x='Sex_Male', y='medalist', data=train)

    # Setting the x and y labels, and setting the legend
    plt.xlabel('Male')
    plt.ylabel('Medalist')
    plt.show()
