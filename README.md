# Email-Campaign-Effectiveness-Prediction-Capstone-Project

# Problem Statement:


Most of the small to medium business owners are making effective use of Gmail-based Email marketing Strategies for offline targeting of converting their prospective customers into leads so that they stay with them in Business. The main objective is to create a machine learning model to characterize the mail and track the mail that is ignored; read; acknowledged by the reader.

Email Campaign Effectiveness Prediction These project is part of the “Machine Learning &Advanced Machine Learning” curriculum as capstone projects at AlmaBetter.

-- Project Status: [Completed]

Objective The main objective is to create a machine learning model to characterize the mail and track the mail that is ignored; read; acknowledged by the reader. Most of the small to medium business owners are making effective use of Gmail-based Email marketing Strategies for offline targeting of converting their prospective customers into leads so that they stay with them in Business.

# Methods Used

Descriptive Statistics

Data Visualization

Machine Learning

Technologies

Python

Pandas

Numpy

Matplotlib

Seaborn

Scikit-learn

XGBoost

# Project Description
EDA - Performed exploratory data analysis on numerical and categorical data.

Data Cleaning - Missing value imputation,Outlier Treaatment

Imabalance handling - First tried Under sampling and implemented baseline models then due to loss of information moved to different technique i.e oversampling using SMOTE and got better results.

Feature Selection - Used information gain for feature selection and dropped features which had less information gain

Model development - Tried different model and finally compared all models F1 and roc_auc score.

# Understanding the Data:

● Email Id : It contains the email id's of the customers/individuals 

● Email Type : There are two categories 1 and 2. We can think of them as marketing emails or important updates, notices like emails regarding the business. 

● Subject Hotness Score : It is the email's subject's score on the basis of how good and effective the content is. 

● Email Source : It represents the source of the email like sales and marketing or important admin mails related to the product. 

● Email Campaign Type : The campaign type of the email. 

● Total Past Communications : This column contains the total previous mails from the same source, the number of communications had. 

● Customer Location : Contains demographical data of the customer, the location where the customer resides. 

● Time Email sent Category : It has three categories 1,2 and 3; the time of the day when the email was sent, we can think of it as morning, evening and night time slots. 

● Word Count - The number of words contained in the email. 

● Total links : Number of links in the email. 

● Total Images : Number of images in the email. 

● Email Status : Our target variable which contains whether the mail was ignored, read, acknowledged by the reader.

# Exploratory Data Analysis:
Exploratory data analysis is a crucial part of data analysis. It involves exploring and analyzing the dataset given to find out patterns, trends and conclusions to make better decisions related to the data, often using statistical graphics and other data visualization tools to summarize the results. 

The visualization tools: Plotly,matplotlib and seaborn. 

# Categorical Insights: 


![image](https://user-images.githubusercontent.com/60994606/153542864-382c17a0-d7f0-402e-94c5-91c62bbb140e.png)

![image](https://user-images.githubusercontent.com/60994606/153542945-742bd310-238c-4003-9091-c11837512810.png)

![image](https://user-images.githubusercontent.com/60994606/153542979-e4d48a86-0760-4929-be02-03eb433200ce.png)

![image](https://user-images.githubusercontent.com/60994606/153543001-05535667-a498-4941-a88c-bd66060fef23.png)

![image](https://user-images.githubusercontent.com/60994606/153543040-76d05a8c-d328-406a-a91d-c198efe8020a.png)

# Observation:
The email type 1 which may be considered as promotional emails are sent more than email type 2 and hence are read and acknowledged more than the other type otherwise the proportion of ignored, read, acknowledged emails are kind of same in both email types.

Email source type shows kind of a similar pattern for both the categories.

In the customer location feature we can find that irrespective of the location, the percentage ratio of emails being ignored, read and acknowledged are kind of similar. It does not exclusively influence our target variable. It would be better to not consider location as a factor in people ignoring, reading or acknowledging our emails. Other factors should be responsible for why people are ignoring the emails not location.

In the Email Campaign Type feature, it seems like in campaign type 1 very few emails were sent but have a very high likelihood of getting read. Most emails were sent under email campaign type 2 and most ignored. Seems like campaign 3 was a success as even when less number of emails were sent under campaign 3, more emails were read and acknowledged.

If we consider 1 and 3 as morning and night category in time email sent feature, it is obvious to think 2 as middle of the day and as expected there were more emails sent under 2nd category than either of the others, sending emails in the middle of the day could lead to reading and opening the email as people are generally working at that time and they frequently check up their emails, but it cannot be considered as the major factor in leading to acknowledge emails.

# Continuous Insights:

![newplot (3)](https://user-images.githubusercontent.com/60994606/153543300-41ad8e37-34a2-471f-bd56-bc945ad77a5c.png)

![newplot (4)](https://user-images.githubusercontent.com/60994606/153543321-d6ef6052-e0de-4ad5-9d3e-1e0c51a0ef6e.png)

![newplot (5)](https://user-images.githubusercontent.com/60994606/153543346-a07e5ae6-9379-4fba-b681-9aaf5d2d4645.png)

![newplot (6)](https://user-images.githubusercontent.com/60994606/153543385-426d4996-93e2-4d87-8baa-d9775e05f93d.png)

![newplot (7)](https://user-images.githubusercontent.com/60994606/153543405-709e2dd8-18ac-4ed6-81b3-7e6883775c9d.png)

# Observation:

In the subject hotness score, the median of ignored emails was around 1 with a few outliers. Acknowledged emails have the most outliers. It is observed that the Subject_Hotness_Score for read and acknowledged emails are much lower.

Analyzing total past communications, we can see that the more the number of previous emails, the more it leads to read and acknowledged emails. This is just about making connections with your customers.

The more the words in an email, the more it has a tendency to get ignored. Too lengthy emails are getting ignored.

The median is kind of similar in all of the three cases in total links feature with a number of outliers.

# Modeling: 
# Logistic Regression: 
Logistic Regression is a classification algorithm that predicts the probability of an outcome that can have only two values. Multinomial logistic regression is an extension of logistic regression that adds native support for multi-class classification problems. Instead, the multinomial logistic regression algorithm is a model that involves changing the loss function to cross-entropy loss and predicting probability distribution to a multinomial probability distribution to natively support multi-class classification problems.
 
# Decision Trees: 
Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems. Decision trees use the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree. Clearly Decision Tree models were overfitting. Both the datasets, whether undersampled or oversampled with SMOTE and SMOTETomek worked really well on train data but not on test data.

# Random Forest: 
Random forests are an ensemble learning method for classification and regression that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. To prevent overfitting, a random forest model was built. Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. 

# KNN Classification: 
K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.

The K-NN algorithm assumes the similarity between the new case/data and available cases and puts the new case into the category that is most similar to the available categories.

K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.

K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.

KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

# XG Boost: 
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. The two reasons to use XGBoost are also the two goals of the project: 

● Execution Speed. 

● Model Performance. 

Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models. XGB SMOTETomek gave the best results till now, with good Test Recall, F1 score and AUC ROC.

More images were there in ignored emails.

There are a considerable number of outliers in Subject_Hotness_Score, Total_Links and Total_Images.
 
# Evaluation Metrics:
 There are a number of model evaluation metrics to choose from but since our dataset was highly imbalanced, it is critical to understand which metric should be evaluated to understand the model performance. 
# Accuracy- 
Accuracy simply measures how often the classifier correctly predicts. We can define accuracy as the ratio of the number of correct predictions and the total number of predictions. Accuracy is useful when the target class is well balanced but is not a good choice for the unbalanced classes, because if the model poorly predicts every observation as of the majority class, we are going to get a pretty high accuracy. Confusion Matrix - It is a performance measurement criteria for the machine learning classification problems where we get a table with a combination of predicted and actual values. 

# Precision - 
Precision for a label is defined as the number of true positives divided by the number of predicted positives. 
# Recall - 
Recall for a label is defined as the number of true positives divided by the total number of actual positives. Recall explains how many of the actual positive cases we were able to predict correctly with our model. 
# F1 Score - 
It's actually the harmonic mean of Precision and Recall. It is maximum when Precision is equal to Recall. 
# AUC ROC - 
The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes. When AUC is 0.5, the classifier is not able to distinguish between the classes and when it's closer to 1,the better it becomes at distinguishing them.
