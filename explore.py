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

def barplot_medalist_male_stat(train):
    # Setting the alpha
    alpha = 0.05

    # Setting the null and alternative hypothesis
    null_hyp = 'Medalists and being male are independent'
    alt_hyp = 'There is a relationship between medalists and being male'

    # Making the observed dataframe
    observed = pd.crosstab(train.Sex_Male, train.medalist)

    # Running the chi square test and running a if statement for the null hypothesis to be rejected or failed to reject
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between medalists and males')

    print('P-Value', p)
    print('Chi2', round(chi2, 2))
    print('Degrees of Freedom', degf)

def barplot_average_age_by_medal(train):
    plt.figure(figsize=(7, 7))
    average_age_by_medal = train.groupby('Medal').mean().reset_index()
    color = ['black','gold','goldenrod','grey']
    sns.barplot(x='Medal', y='Age', data=average_age_by_medal, order=average_age_by_medal.sort_values('Age').Medal, palette=color)
    plt.ylim(22)
    plt.ylabel('Average Age')
    plt.title('Average Age By Medal')

def athlete_most_medals(df):
    groups = df.groupby('Name').sum()
    groups[groups['medalist'] == groups['medalist'].max()]

def agebin_most_medals(df):
    groups = df.groupby('AgeBins').sum()
    groups[groups['medalist'] == groups['medalist'].max()]
