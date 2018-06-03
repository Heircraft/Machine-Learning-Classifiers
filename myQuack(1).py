
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.


'''
import numpy as np;
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
    


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
    #open the file and read in all the data
    fl = open(dataset_path, "r");
    
    line = fl.readline();
    rows = [];
    while line:
        splitline = line.split(",");
        rows.append(splitline);
        line = fl.readline();
       
    rowsArr = np.array(rows);
    np.random.shuffle(rowsArr);

    y1 = [];
    X1 = [];
    n = rowsArr.shape[0];
    #iterate through the file and assign all the data
    for x in range(n):
        X1.append(rowsArr[x][2:]);
        Y1 = rowsArr[x][1];
        if Y1 == 'M':
            y1.append(1);
        else:
            y1.append(0);
     
    y = np.array(y1);
    X = np.array(X1);
    
    #split the data 90% training and 10% testing
    n9 = int(n*0.9);
    Xtrain, Xtest = X[:n9], X[n9:];
    ytrain, ytest = y[:n9], y[n9:];
    
    return Xtrain, ytrain, Xtest, ytest;
    

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
    #fit the data to the GaussianNB() classifier
    X_training = X_training.astype(np.float);
    clf = GaussianNB();
    clf.fit(X_training, y_training);    
    #return the mean of 10 cross validated scores
    scores = cross_val_score(clf, X_training, y_training, cv=10, scoring='accuracy').mean();
    print("Cross validated accuracy: ", scores) 
    
    return clf;

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
    #the range of values used to find the best value for our hyperparameter
    d_range = range(1, 20);
    d_scores = [];
    
    #iterate over range and cross validate 10 times each iteration
    for d in d_range:
        #set random state to ensure a constant randomness
        clf = DecisionTreeClassifier(max_depth =d, random_state=20);
        scores = cross_val_score(clf, X_training, y_training, cv=10, scoring='accuracy').mean();
        
        if scores.mean() >= max(d_scores + [0]):
            best_d = d;
        
        d_scores.append(round(scores.mean(), 5));
    
    #plot
    plt.scatter(d_range, d_scores)
    z = np.polyfit(d_range, d_scores, 4)
    p = np.poly1d(z)
    plt.plot(d_range,p(d_range),"r--")    
    plt.xlabel("Value of max_depth for DT")
    plt.ylabel("Cross-validated accuracy")
    plt.show();
    
    # best_d (depth) is the hyperparameter
    print("Best D value = ", best_d);
    
    #take our best parameter and input it in our classifier
    dtf = DecisionTreeClassifier(max_depth = best_d, random_state = 10);
    dtf.fit(X_training, y_training)
    
    #return the mean of 10 cross validated scores to find cross validated accuracy
    score = cross_val_score(dtf, X_training, y_training, cv=10, scoring='accuracy').mean();
    
    print("Cross-validated score with best d: ", score);
    print("\nStandard Deviation of d_scores: ", np.std(d_scores));
    print("\n");
    
    return dtf;

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
    
    #the range of values used to find the best value for our hyperparameter
    k_range = range(1, 51);
    k_scores = [];
    
    #iterate over range and cross validate 10 times each iteration
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k);
        scores = cross_val_score(clf, X_training, y_training, cv=10, scoring='accuracy');
        
        if scores.mean() >= max(k_scores + [0]):
            best_k = k
        
        k_scores.append(round(scores.mean(), 5));
    
    #plot
    plt.scatter(k_range, k_scores)
    z = np.polyfit(k_range, k_scores, 3)
    p = np.poly1d(z)
    plt.plot(k_range,p(k_range),"r--")    
    plt.xlabel("Number of neighbours for KNN")
    plt.ylabel("Cross-validated accuracy")
    plt.show();
    
    # best_k is the hyperparameter
    print("Best K value = ", best_k);
    
    #take our best parameter and input it in our classifier
    knn = KNeighborsClassifier(n_neighbors=best_k);
    knn.fit(X_training, y_training)
    
    #return the mean of 10 cross validated scores to find cross validated accuracy
    score = cross_val_score(knn, X_training, y_training, cv=10, scoring='accuracy').mean();
    
    print("Cross-validated score with best K: ", score);
    print("\nStandard Deviation of k_scores: ", np.std(k_scores));
    print("\n");
    
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

    #the range of values used to find the best value for our hyperparameter
    g_range = np.logspace(-6, -1, 5);
    g_scores = [];
    
    #iterate over range and cross validate 10 times each iteration
    for g in g_range:
        clf = SVC(gamma=g);
        scores = cross_val_score(clf, X_training, y_training, cv=10, scoring='accuracy');
     
        if scores.mean() >= max(g_scores + [0]):
            best_g = g
     
        g_scores.append(round(scores.mean(), 5));
    
    #plot
    plt.semilogx(g_range, g_scores, label="Training score",color="darkorange", lw=2)
    plt.xlabel("Value of gamma for SVM")
    plt.ylabel("Cross-validated SVM")
    plt.show();
    
    # best_g (gamma) is the hyperparameter
    print("Best g value = ", best_g);
    
    #take our best parameter and input it in our classifier
    svm = SVC(gamma=best_g);
    svm.fit(X_training, y_training)
    
    #return the mean of 10 cross validated scores to find cross validated accuracy
    score = cross_val_score(svm, X_training, y_training, cv=10, scoring='accuracy').mean();
     
    print("Cross-validated score with best g: ", score);
    print("\nStandard Deviation of g_scores: ", np.std(g_scores));
    print("\n");
    
    return svm;
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass

    # Load the dataset and split
    x_training, y_training, x_test, y_test = prepare_dataset("medical_records(1).data");
    
    print("===============================================================");
    # build Naive Bayes classifier
    print("NAIVE BAYES");
    clf = build_NB_classifier(x_training, y_training);
    x_test_NB = x_test.astype(np.float);
    x_training_NB = x_training.astype(np.float);
    # Training prediction error
    training_error = 1 - clf.score(x_training_NB, y_training);
    print("Training Error (BAY):",training_error);
    # Testing prediction error
    accuracy = clf.score(x_test_NB, y_test);
    testing_error = 1 - clf.score(x_test_NB, y_test);
    print("Testing error (BAY)", testing_error);
    print("Testing accuracy (BAY)", accuracy);
    print("===============================================================");
    
    # Build DT classifier
    print("DECISION TREE");
    clf2 = build_DT_classifier(x_training, y_training);
    # Training prediction error
    training_error = 1 - clf2.score(x_training, y_training);
    print("Training Error (DT):",training_error);
    # Testing prediction error
    accuracy = clf2.score(x_test, y_test);
    testing_error = 1 - accuracy;
    print("Testing Error (DT)",testing_error);
    print("Testing accuracy (DT)", accuracy);
    print("===============================================================");
    
    # Build NN classifier
    print("NEAREST NEIGHBOUR");
    clf3 = build_NN_classifier(x_training, y_training);
    # Training prediction error
    training_error = 1 - clf3.score(x_training, y_training);
    print("Training Error(KNN):",training_error);
    # Testing prediction error
    testing_error = 1 - clf3.score(x_test, y_test);
    print("Testing Error(KNN)",testing_error);
    print("===============================================================");
    
    # Build SVM classifier
    print("SUPPORT VECTOR MACHINE");
    clf3 = build_SVM_classifier(x_training, y_training);
    # Training prediction error
    training_error = 1 - clf3.score(x_training, y_training);
    print("Training Error (SVM):",training_error);
    # Testing prediction error
    accuracy = clf3.score(x_test, y_test);
    testing_error = 1 - accuracy;
    print("Testing Error (SVM)",testing_error);
    print("Testing accuracy (SVM)", accuracy);
    print("===============================================================");
    

