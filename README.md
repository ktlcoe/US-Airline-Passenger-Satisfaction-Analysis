# US-Airline-Passenger-Satisfaction-Analysis
Problem Statement
As the world begins to see the possible end to the devastating pandemic that has permeated the global and the start of a new normal, people are beginning to feel safe enough to travel again. Airlines have suffered greatly over the past year as their passengers chose to quarantine and remain grounded for their own safety. However, as vaccine distribution increases and passengers begin to feel safe enough to travel again, airlines must maximize on their marketing and passenger experience in an attempt to recover their recent losses as quickly as possible.

Understanding what predictors most impact customer satisfaction as well as the right demographics to target for marketing could be significant contributors to the speed of an airline’s recovery. Data is one of our greatest assets and analyzing data surrounding customer satisfaction could aid in such efforts to understand the best approach for an airline to pursue. Airlines are really starting to dive into data analytics to understand what benefits can be extracted to turn information into action that could improve their passengers’ experience and of course improve their profit [5].

This project will utilize passenger satisfaction data to determine the important predictors of passenger satisfaction so that airlines can target their improvements to best serve the passenger experience. Those predictors will then be investigated for use to tune per demographic to further understand what predictors would improve passenger satisfaction for a particular demographic group.

Data Source
	US Passenger Airline Satisfaction (129,880 satisfaction ratings). This dataset is from Kaggle: https://www.kaggle.com/johndddddd/customer-satisfaction

Proposed Methodology
This project will create a logistic regression model and use tree based methods to determine the best model to predict customer satisfaction given the predictors in the dataset. The predictors will also be evaluated using dimensionality reduction to address the proposed problem of an airline for where to focus their resources to maximize their marketing and improve customer experience.

Other potential algorithms that will be considered for use in predicting the customer satisfaction include Support Vector Machines and K-Nearest-Neighbors. As part of the analysis of the results, the different predictive accuracy of each method will be examined and evaluated.
 
Logistic Regression
The logistic regression linear model can be used to assume a classification of data from a statistical model. The predicted probabilities for a class in a Logistic regression model must sum to be one and remain within a domain of 0 and 1 [3]. The binary prediction of a satisfied customer versus an unsatisfied customer would be a two-class prediction, for which the logistic regression model is aptly suited. 

The logistic regression model can be optimized to maximize the conditional likelihood of a customer belonging to the class of satisfied or unsatisfied. For example, the probability of a customer being satisfied as defined by P(y=1| x,θ) can be given by the following equation:

P(y=1| x,θ) =1/(1 + exp(-θ^T x)) 

To solve the parametric equation, the log-likelihood equation is used in conjunction with the gradient descent to maximize the conditional likelihood of a customer belonging to the class of satisfied.
Tree Based Methods
Classification Trees are a commonly used supervised learning method for classifying observations. While single trees provide a strong visual based interpretation, they often suffer from poor predictive accuracy and generalization [4].

A generalization of classification trees is an algorithm called Random Forests [1] which seeks to improve accuracy on unseen data though the construction of numerous trees using bootstrapped data where only a subset of randomly selected features are considered at each split. This can result in a highly flexible model, with strong “out of the box” results. However, due to the multitude of trees produced, the interpretation of which features are most important can be more challenging than linear models such as logistic regression.

Random Forests will be used within this project to predict whether or not airline customers will be satisfied with their flight.

Dimensionality Reduction
When working with multivariate data, dimensionality reduction techniques are frequently used to reduce the dangers of overfitting. Principal component analysis, or PCA, is one such technique. PCA is specifically a type of feature extraction, meaning that new variables are created that each contain information from all of the original variables. These newly created components will be orthogonal to every other component, thus ensuring independence between variables which is often a requirement for modeling. Any number of those newly created variables can then be used in the modeling process, as they are ordered by importance [2].
The airline satisfaction data used in this project has 22 potential predictors. Initial analysis shows potential multicollinearity, especially between Departure Delay and Arrival Delay. As such, PCA will be considered as part of the project. All previously discussed models will also be attempted on the standard dataset and then after PCA reduction for comparison. Further tuning will be done on the number of PCA components used in the modeling to ensure the best performance.
Hypothesized Results
By passenger experience rationale, it is expected that Departure Delay and Arrival Delay will be important predictors of customer satisfaction. As noted in discussion of the predictor dimensionality reduction, the two predictors indicate possible multicollinearity so it is likely one one will be used during the implementation of the reduction method. 

It could also be expected to see some coalescence around correlation of important predictors and demographic markers such as age and gender. For example, younger customers may have higher correlation between satisfaction and Wifi service whereas older customers have higher correlation between satisfaction and seat comfort. Airlines could in turn use the demographic information they have for their customers and create targeted marketing campaigns surrounding such findings.
