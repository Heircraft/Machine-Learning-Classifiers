
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.


'''
import numpy as np;
import random;
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GuassianNB


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9684026, 'Alexander', 'Santander'), (9654780, 'Hrushikesh', 'Lakkola')]
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"   
    # validation in classifier

    fl = open(dataset_path, "r");
    
    line = fl.readline();
    rows = [];
    while line:
        splitline = line.split(",");
        rows.append(splitline);
        line = fl.readline();
       
    rowsArr = np.array(rows);
    np.random.shuffle(rowsArr);
    print(len(rowsArr));

    y1 = [];
    X1 = [];
    n = rowsArr.shape[0];
    for x in range(n):
        X1.append(rowsArr[x][2:]);
        Y1 = rowsArr[x][1];
        if Y1 == 'M':
            y1.append(1);
        else:
            y1.append(0);
     
    y = np.array(y1);
    X = np.array(X1);

    n9 = int(n*0.9);
    Xtrain, Xtest = X[:n9], X[n9:];
    ytrain, ytest = y[:n9], y[n9:];
    
    return Xtrain, ytrain, Xtest, ytest;
    
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    clf = GuassianNB();
    
    clf.fit(X_training, y_training);
    
    
    
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    k_range = range(1, 51);
    
    k_scores = [];
    
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k);
        scores = cross_val_score(clf, X_training, y_training, cv=10, scoring='accuracy');
        
        if scores.mean() >= max(k_scores + [0]):
            best_k = k
        
        k_scores.append(round(scores.mean(), 5));
    
    plt.plot(k_range, k_scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross-validated accuracy")
    plt.show();
    
    # best_k is the hyperparameter
    print("Best K value = ", best_k);
    
    knn = KNeighborsClassifier(n_neighbors=best_k);
    
    knn.fit(X_training, y_training)
    
    score = cross_val_score(knn, X_training, y_training, cv=10, scoring='accuracy').mean();
    
    print("Cross-validated score with best K: ", score);
    
    return knn;
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    # linearsvc is same as kernel=linear Go with JUST SVC HOE
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # call your functions here
    # Load the dataset and split
    x_training, y_training, x_test, y_test = prepare_dataset("medical_records(1).data");
    
    # build Naive Bayes classifier
    
    # Build NN classifier
    clf = build_NN_classifier(x_training, y_training);
    
    # Training prediction error
    training_error = 1 - clf.score(x_training, y_training);
    print(training_error);
    # Testing prediction error
    testing_error = 1 - clf.score(x_test, y_test);
    print(testing_error);
    
    
    


