
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.


'''
import numpy as np;



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
    line = fl.readline()
    line = fl.readline()
    cnt = 0
    rows = [];
    while line:
        line = fl.readline();
        splitline = line.split(",");
        rows.append(splitline);
        #print(rows[cnt])
        cnt = cnt + 1;
       
    rowsArr = np.array(rows)
    
    n = rowsArr.shape[0]
    n9 = int(n*0.9);
    p = np.random.permutation(n);
    #X, y = rowsArr[:][]
    
    for x in range(n-1):
        X = rowsArr[x][2:];
        print(rowsArr[x][1]);
                      
#        if (rowsArr[x][1] == M) {
#            rowsArr[x][1] = 1;
#        } else {
#            rowsArr[x][1] = 0;
#        }
    
        y = rowsArr[x][1];
#    print(y);           
    print(rowsArr[x][1]);
    
#    X = [];
#    y = [];
#    line = fl.readline()
#    cnt = 0
#    while line:
#        #print("Line {}: {}".format(cnt, line.strip()))
#        splitLine = line.split(",");
#        ID = splitLine[0];
#        cLabel = splitLine[1];
#        rValued = splitLine[2:];
#        
#        if cLabel is 'M':
#            y.append(1);
#        else:
#            y.append(0);
#            
#        X.append(rValued);
#        
#        line = fl.readline()
#        cnt += 1;
#    
#    Xret = np.array(X);
#    yret = np.array(y);
#    
##    print(Xret);
##    print("\n");
##    print(yret);
##    print("------------");
#
#    return y and X;
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
    raise NotImplementedError()

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
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # call your functions here
    #y_training, x_training = prepare_dataset("medical_records(1).data");
    prepare_dataset("medical_records(1).data");
    #build_NB_classifier(x_training, y_training);
    


