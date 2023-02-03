# CustomModels
A compilation of the work done in my Machine Learning module. A naive attempt to replicate the behavior of built in MATLAB machine learning models. All the fit functions work identical to the inbuilt MATLAB fit functions.

# Models
I made only a small selection of models with limited hyperparameters as these models were all we covered on the course. Below are listed the models and their hyperparameters:
+ Naive Bayes
+ KNN
  - NumNeighbours
  - K-d tree or exaustive search
+ Decision Tree
  - MaxNumSplits
  - MinParentSize
+ Perceptron (Net with 0 hidden layers)
  - Learning Rate
  - Function
+ Ensemble
  - Method
    - Soft Vote
    - Sub Space
    - Bagging
  - Learners
    - Naive Bayes
    - KNN
    - Decision Tree
  - NumLearningCycles
  - NFeaturesToSample

# Using The Project
## Using Models
Below shows how to use my models. They are exactly the same as the default
```matlab
% make knn model with training data
m_knn = my_fitcknn(train_examples, train_labels, 'NumNeighbours', 3)

predictions = m_knn.predict(test_examples)
preformance = sum(predictions == train_labels)/height(train_labels)
```

## Using Ensemble
The ensembles also work the same as the default.
```matlab
% ensamble with NB and KNN
m_en = my_fitcensemble(train_examples,train_labels, ...
    'Learners', {my_templateKNN('NumNeighbors',5), my_templateNB});
  
predictions = m_en.predict(test_examples)
preformance = sum(predictions == train_labels)/height(train_labels)
```

# Tests
The test file shows the use of all the models aswell as an abundance of information. It has been included in a seperate folder to the models, so if you want to run the live scrips you must first move all of the files from the 'Test' folder to the main folder of models.
