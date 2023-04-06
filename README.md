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
This projects goal is to predict what athlets will medal in the olympics using a classification model. 

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
 - Webscraping/acquiring more information.
 - Testing more models.


### Steps to Reproduce:
- Then download the wrangle.py, model.py, explore.py, and final_report.ipynb
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
- Featured engineered 4 new columns medalist, Sex_Male, BMI, and AgeBins.
- Split the data into train, validate, and test.
- Imputed age with the mean.
- Dropped all data before the year 1960 as there is a large number of nulls.
- Filled the nulls in the Medal column with the string 'None'.
- Dropped the remaining nulls in the dataset.

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - model.py
    - explore.py


### Takeaways from exploration:
- 

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


***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline
    
- Baseline Results: Accuracy 86%
    

- Selected features to input into models:
    - features = ['Sex_Male', 'Age', 'BMI']

***

## Models:


### Model 1: Decision Tree


Model 1 results:
- Decision Tree max_depth=7, max_features=2, min_samples_leaf=4, criterion="entropy"
- Model stats:
- Accuracy: 0.86
- True Positive Rate: 0.00
- False Positive Rate: 0.00
- True Negative Rate: 1.00
- Flase Negative Rate: 1.00
- Precision: nan
- Recall: 0.00
- f1 score: nan
- Positive support: 14949
- Negative support: 90465
- Accuracy of Decision Tree classifier on training set: 0.86



### Model 2 : K-Nearest Neighbor


Model 2 results:
- KNeighborsClassifier leaf_size=2, n_neighbors=6, weights="uniform"
- Model stats:
- Accuracy: 0.86
- True Positive Rate: 0.01
- False Positive Rate: 0.01
- True Negative Rate: 0.99
- Flase Negative Rate: 0.99
- Precision: 0.27
- Recall: 0.01
- f1 score: 0.02
- Positive support: 14949
- Negative support: 90465
- Accuracy of KNN classifier on training set: 0.86

### Model 3 : Naive Bayes

Model 3 results:
- Model stats:
- Accuracy: 0.86
- True Positive Rate: 0.00
- False Positive Rate: 0.00
- True Negative Rate: 1.00
- Flase Negative Rate: 1.00
- Precision: 0.26
- Recall: 0.00
- f1 score: 0.00
- Positive support: 14949
- Negative support: 90465
- Accuracy of Naive Bayes classifier on training set: 0.86


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation |
| ---- | ----|
| Baseline | 0.86 |
| Decision Tree | 0.86 |
| K-Nearest Neighbor | 0.86 | 
| Naive Bayes | 0.86 |


- {K-Nearest Neighbor} model performed the best.


## Testing the Model

- Model Testing Results: 81% Accuracy

***

## <a name="conclusion"></a>Conclusion:

- 

#### Idealy we would like to find a way to incentivise more people to sign a one or two year contract as opposed to a month-to-month contract. It would also be benificial to get more people to stay past the average tenure and keep their cost at or below the average monthly charge. 

[[Back to top](#top)]
