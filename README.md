# ml-stress-detection
Maxime Kotur - CSCI Neural Networks Project Proposal

1	Analysis
1.1	Problem Description:
Depression is becoming increasingly common among adolescents (Miller & Campo, 2021), and it is still challenging to self-detect it. While many schools and companies offer counseling at no cost or a reduced cost, many people struggle with making the decision to see a doctor for depression.
Furthermore, the global COVID outbreak led to a significant increase in depression and anxiety rates, especially in females (Hawes et al.,2021). However, machine learning algorithms can help individuals make the decision whether they need to see a doctor for depression through simple sets of questions that require open-ended input from individuals.
Problem: Can individuals self-detect depression and decide to see a doctor for formal diagnosis and treatment?
1.2	Performance Criteria:
●	ROC AUC (or F1 scores) will be used as the main metric. (Depending on the data - balanced/imbalanced)
●	Confusion matrix
●	0.8 success rate is aimed.

1.3	Related Work:
Kipli, Kouzani and Hamid (2013) used Support Vector Machines and IG-Random Forest methods to detect depression and they have obtained 85.23% accuracy by combining these two methods.
Similarly, the Boruta algorithm and Random Forest classifier were used by Haque, Kabir and Khanam (2021) to detect child depression. He/she found that the RF performed far better than other machine learning methods in child depression detection with 95 % accuracy and 99 % precision rate.
Furthermore, some hands-on projects built with the same dataset by other engineers/researchers provide an assessment for a single input only, while this project aims to ask users multiple questions for textual input that will be relevant to the dataset and take the average of the prediction for each question. In model-level, this model will be utilizing KNN and logistic regression and compare their results while other engineers/researchers mainly used LSTM and Naïve-Bayes with this dataset.
1.2 Project Objective:
The goal of the project is to create a simple-to-use depression detector model that provides highly accurate results for at least 80% user input.
 
2	Hypothesis
Data: The data that will be used in this project can be accessed via; https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned. The dataset has two columns ( clean_text (input) and is_depression). There are 7650 rows. The texts are classified as 0(no depression) and 1 (depression). The data does not include any confidential information or inappropriate data.
Methods: As the data comes labeled already, we will be using supervisor algorithms. We are planning to use k-nearest neighbors and logistic regression methods and compare their performance on the test data.Logistic regression is easy to implement and very efficient in data training. With minimal effort, it can yield decent performance as a linear method. On the other hand, KNN is salso easy to implement and generally provides highly accurate results with flexible distance choice as a non-linear method. These two methods seem to be more appropriate for our data set for binary classification.
Experiment: We will be evaluating the results based on F1 scores or ROC AUC. The desired success rate is 0.8.

3	Synthesis
The software/ML libraries that will be used as follows; scikitlearn, numpy, matplotlib, pandas, tensorflow, keras, spacy, streamlit or react.js, python, google collab.
For KNN, n_neighbors will be the main HP and only odd numbers will be used. Distance metric will also be used to test out different metrics such as Manhattan and Euclidian.
For logistic regression, solver, reg strength and penalty HPs will be tuned.
Phase 1: Data familiarization and cleanup, defining data variables and splitting data by 80/20 Phase 2: Data transformations on test and train after the split to avoid leakage
Phase 3: implementing KNN model, HP tuning and and Performance Measurement Phase 4: Implementing LR model, HP tuning and Performance Measurement Phase 5: Model-by-model Comparison of Predicted Results
Phase 6: *Creating a minimum of 5 questions from the dataset to ask for user input
Phase 7: Assessing Overall Performance by taking the average of predictions for each question (Depressed or Not )
Phase 8: Building a React or Streamlit Web App and interacting with users.

4	Validation
4.1	Results:
ROC AUC score will be mainly used for assessment, and 0.8 score will be required. Tensorboard summary will be included for results with different parameters.
 
4.2	Conclusions:
Formal Conclusions: Kolmogorov-Smirnov test results will be used to provide formal conclusions. P-values (<0.05) will be provided.
REFERENCES
Haque, U. M., Kabir, E., & Khanam, R. (2021). Detection of child depression using machine learning methods. PLoS one, 16(12), e0261131.

Hawes, M. T., Szenczy, A. K., Klein, D. N., Hajcak, G., & Nelson, B. D. (2021). Increases in depression and anxiety symptoms in adolescents and young adults during the COVID-19 pandemic. Psychological Medicine, 1-9.

Kipli, K., Kouzani, A. Z., & Hamid, I. R. A. (2013). Investigating machine learning techniques for detection of depression using structural MRI volumetric features. International journal of bioscience, biochemistry and bioinformatics, 3(5), 444-448.

Miller, L., & Campo, J. V. (2021). Depression in adolescents. New England Journal of Medicine, 385(5), 445-449.



Clarification Questions
1.	To our knowledge, the final decision regarding finding the efficient ML methods for this problem will be found based on the data points in the dataset. However, I would like to ask if it makes sense to use Decision Trees/Random Forest in a binary classification problem. I believe these methods would make more sense to use in multi-class classification problems, but I am also curious if it would be possible to use them for binary classification,?


2.	I am currently planning to work on a binary classification problem, but I am curious if it would be possible to convert a binary classification problem into a multi-class classification problem such as “Not depressed, slightly depressed, depressed, very depressed “ etc instead of “Depressed, Not depressed”. To our knowledge, to be able to perform this conversion, our dataset must be labeled accordingly (0/1/2/3) rather than simply 0/1. Would working on this be a good practice and challenge and do you think this is something we would be able to handle?


3.	I aim to design this model and application in a way that end-users can easily interact with the model. To do this, I plan to ask users for their input through 5-10 open-ended questions based on the dataset that we trained. However, I have two challenges here. The first one is how to specify these questions. Is there a statistical way to specify these questions? The second is the overall assessment. The current plan aims to assess the input for every question separately, label them (depressed or not depressed) and then take the average of these labels to come up with an overall conclusion. However, this sounds like a very rough prediction, and we were wondering if we can come up with better ways to detect overall depression.
