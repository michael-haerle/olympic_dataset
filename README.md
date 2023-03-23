# <a name="top"></a>Medals at the Olympics (PLACEHOLDER README)
![]()

by: Michael Haerle

<p>
  <a href="https://github.com/michael-haerle" target="_blank">
    <img alt="Michael" src="https://img.shields.io/github/followers/michael-haerle?label=Follow_Michael&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___


## <a name="project_description"></a>Project Description:


[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
- Create README.md with data dictionary, project and business goals, come up with questions to lead the exploration and the steps to reproduce.
- Acquire and prepare data for exploration from Kaggle and create a function to automate this process. Save the function in an wrangle.py file to import into the Final Report Notebook.
- Produce at least 4 clean and easy to understand visuals.
- Clearly define hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy.
- Train three different classification models.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.


### Project goals: 
- To predict which athletes are going to medal in the olympics.


### Target variable:
- medalist

### Initial questions:
- What are the correlations in the data?
- What athlete has the most medals?
- What age bin has the most medals?
- Who are the top performing teams? What team has the most medals?
- Is there a relationship between medalists and males?
- Is there a significant difference in the average age for different medals?

### Need to haves (Deliverables):
- A final report notebook

### Nice to haves (With more time):
 - 


### Steps to Reproduce:
- Then download the wrangle.py, model.py, explore.py, telcoCo.png, and final_report.ipynb
- Make sure these are all in the same directory and run the final_report.ipynb.

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- 


***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| ID | Athletes unique number| int64 |
| Name | Athletes name | object |
| Sex | Athletes gender M = Male, F = Female | object |
| Age | Athletes age | float64 |
| Height | Athletes height in centimeters | float64 |
| Weight | Athletes weight in kilograms | float64 |
| Team | Athletes team name | object |
| NOC | National Olympic Committee 3 letter code | object |
| Games | Year and season | object |
| Year | Year the games took place | int64 |
| Season | Summer or Winter games | object |
| City | Host city | object |
| Sport | Sport played | object |
| Event | Specific event in the sport | object |
| Medal | What medal the athlete received if any| object |
| Sex_Male | 1 = Athlete is male, 0 = Athlete is female | int64 |
| medalist | 1 = Athlete received a medal, 0 = Athlete received no medal | int64 |
| BMI | Athletes Body Mass Index | float64 |
| AgeBins | Athletes age binned by every 10 years | category |

***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Prepare steps: 
- Droped duplicate columns
- Created dummy variables
- Concatenated the dummy dataframe
- Changed the type for senior_citizen, tenure, and monthly_charges
- Mapped the yes and no to 1 and 0 for columns partner, dependents, phone_service, paperless_billing, and churn
- Dropped columns not needed
- Changed the type of total_charges by replacing the white space with a 0
- Set variables for the mean of tenure and monthly charges
- Used those variables to feature engineer a new column where it returned a true if they were above the average monthly charge and below the average tenure
- Mapped the true and false of that columns to 1 and 0
- Split into the train, validate, and test sets

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - model.py
    - explore.py


### Takeaways from exploration:
- The new column I made for people above the Avg Monthly Charge and Below the Avg Tenure seems to be a good column to use in the modeling phase.
- Most of the numeric columns will be used for the modeling phase.

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: Contract Type and churn are independent
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between churn and Contract Type

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- We reject the null hypothesis that Contract Type and churn are independent
- There is a relationship between churn and Contract Type
- P-Value 3.2053427834370596e-153
- Chi2 702.26
- Degrees of Freedom 2


### Stats Test 2: Chi Square


#### Hypothesis(First Plot):
- The null hypothesis (H<sub>0</sub>) is: Phone Service and churn are independent
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between churn and Phone Service

#### Confidence level and alpha value(First Plot):
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results(First Plot):
- We fail to reject the null hypothesis that Phone Service and churn are independent
- There appears to be no relationship between churn and Phone Service
- P-Value 0.11104986402814591
- Chi2 2.54
- Degrees of Freedom 1

#### Hypothesis(Second Plot):
- The null hypothesis (H<sub>0</sub>) is: Females who have Phone Service and churn are independent
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between churn and Females with Phone Service

#### Confidence level and alpha value(Second Plot):
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results(Second Plot):
- We reject the null hypothesis that Females who have Phone Service and churn are independent
- There is a relationship between churn and Females with Phone Service
- P-Value 0.0298505787547087
- Chi2 4.72
- Degrees of Freedom 1


### Stats Test 3: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: People who are above the Avg Monthly Charge and Below the Avg Tenure are independent with churn
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between churn and people who are above the Avg Monthly Charge and Below the Avg Tenure

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- We reject the null hypothesis that People who are above the Avg Monthly Charge and Below the Avg Tenure are independent with churn
- There is a relationship between churn and people who are above the Avg Monthly Charge and Below the Avg Tenure
- P-Value 5.788583939458989e-132
- Chi2 597.52
- Degrees of Freedom 1


***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline
    
- Baseline Results: Accuracy 73%
    

- Selected features to input into models:
    - features = ['bel_avg_ten_abv_avg_mon_chrg', 'internet_service_type_None', 'internet_service_type_Fiber_optic', 'contract_type_Two_year', 'contract_type_One_year', 'gender_Male', 'monthly_charges', 'paperless_billing', 'tenure', 'dependents', 'partner', 'senior_citizen']

***

## Models:


### Model 1: Random Forest


Model 1 results:
- RandomForestClassifier min_samples_leaf=12, max_depth=8, random_state=123
- Model stats:
- Accuracy: 0.81
- True Positive Rate: 0.51
- False Positive Rate: 0.08
- True Negative Rate: 0.92
- Flase Negative Rate: 0.49
- Precision: 0.71
- Recall: 0.51
- f1 score: 0.59
- Positive support: 1121
- Negative support: 3104
- Accuracy of random forest classifier on training set: 0.81



### Model 2 : K-Nearest Neighbor


Model 2 results:
- KNeighborsClassifier n_neighbors=15
- Model stats:
- Accuracy: 0.81
- True Positive Rate: 0.51
- False Positive Rate: 0.08
- True Negative Rate: 0.92
- Flase Negative Rate: 0.49
- Precision: 0.69
- Recall: 0.51
- f1 score: 0.58
- Positive support: 1121
- Negative support: 3104
- Accuracy of KNN classifier on training set: 0.81

### Model 3 : Logistic Regression

Model 3 results:
- LogisticRegression C=.01, random_state=123, intercept_scaling=1, solver=lbfgs
- Model stats:
- Accuracy: 0.80
- True Positive Rate: 0.46
- False Positive Rate: 0.09
- True Negative Rate: 0.91
- Flase Negative Rate: 0.54
- Precision: 0.66
- Recall: 0.46
- f1 score: 0.55
- Positive support: 1121
- Negative support: 3104
- Accuracy of Logistic Regression classifier on training set: 0.80


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation |
| ---- | ----|
| Baseline | 0.73 |
| Random Forest | 0.80 |
| K-Nearest Neighbor | 0.79 | 
| Logistic Regression | 0.80 |


- {Random Forest} model performed the best


## Testing the Model

- Model Testing Results: 81% Accuracy

***

## <a name="conclusion"></a>Conclusion:

- There are positive correlations with churn and monthly charges, papperless billing, fiber optic, phone service, senior citizen, and those who are above the average monthly charges and below the average tenure.
- There are negative correlations with churn and tenure, total charges, no internet service, and a two year contract.
- There is a relationship between churn and contract type.
- There was not a relationship between both genders who churn and phone service, however there was a relationship between females who churn and phone service.
- There was a relationship between churn and people who are above the average monthly charge and below the average tenure.

#### Idealy we would like to find a way to incentivise more people to sign a one or two year contract as opposed to a month-to-month contract. It would also be benificial to get more people to stay past the average tenure and keep their cost at or below the average monthly charge. 

[[Back to top](#top)]
