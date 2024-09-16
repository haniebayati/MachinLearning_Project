# import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_validate, cross_val_predict

# function to read excel file and return a column
def read_excel_column(fileName, columnName):
        # read excel file
        data = pd.read_excel(fileName)
        # check whether the column exists in Excel or not
        if columnName not in data.columns:
            raise ValueError(f"Column '{columnName}' not found in the Excel file.")
        # get data column
        column_data = data[columnName].astype(str) # Make sure the column is treated as a string 
        return column_data
   
# Function to check if a string contains a phone number using regex
def contains_number(text):
    # Regular expression to match phone numbers in different formats
    phone_pattern = r'(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})'
    return re.search(phone_pattern, text) is not None

# Function to check if a string contains a link using regex
def contains_link(text):
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.search(link_pattern, text) is not None

# Function to check for content styling (e.g., excessive capital letters, unusual punctuation)
def has_content_styling(text):
    # Example: Check for all caps or excessive exclamation/question marks
    if text.isupper() or re.search(r'[!]{2,}|\?{2,}', text):
        return True
    return False

# Function to culculate count of pronouns 
def count_pronouns(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Perform part-of-speech tagging on the words
    pos_tags = nltk.pos_tag(words)
    # Define the POS tags corresponding to pronouns
    # Personal pronouns (PRP) and Possessive pronouns (PRP$)
    pronoun_tags = {'PRP', 'PRP$'}
    # Count the pronouns in the text
    pronouns_count = Counter(tag for word, tag in pos_tags if tag in pronoun_tags)
    # Get the total by summing the counts from the Counter object and return it
    return sum(pronouns_count.values())

# Function to culculate length of string 
def culculate_length(text):
    text_length = len(text)
    return(text_length)
    
      
# give a value to columnName and fileName and call read_excel_column function
fileName = r'HW2_AUT_MLPR_4021-1-SPAM text message 20170820.xlsx'
columnName = 'Message' 
excel_column = read_excel_column(fileName, columnName)

# only continue if exel_column is not empty
if excel_column is not None:
    
    link_result1 = []
    phone_number_result1 = []
    content_styling_result1 = []
    count_of_pronouns1 = []
    culculate_of_length1 = []
    
    # Downloading the necessary NLTK datasets
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    # for each row in excel_column:
    for row in excel_column: 
        
        # Functions to construct feature vectors are invoked
        link_result = 1 if contains_link(row) else 0
        phone_number_result = 1 if contains_number(row) else 0
        content_styling_result = 1 if has_content_styling(row) else 0
        count_of_pronouns = count_pronouns(row)
        culculate_of_length = culculate_length(row)
        
        # add results to lists to build feature vectors
        link_result1.append(link_result)
        phone_number_result1.append(phone_number_result)
        content_styling_result1.append(content_styling_result)
        count_of_pronouns1.append(count_of_pronouns)
        culculate_of_length1.append(culculate_of_length)
    
    # Convert each list into a DataFrame and give a column name 
    link_result12 = pd.DataFrame(link_result1, columns=['contain_link'])
    phone_number_result12 = pd.DataFrame(phone_number_result1, columns=['has_phone_number'])
    content_styling_result12 = pd.DataFrame(content_styling_result1, columns=['content_styling'])
    count_of_pronouns12 = pd.DataFrame(count_of_pronouns1, columns=['pronouns_count'])
    culculate_of_length12 = pd.DataFrame(culculate_of_length1, columns=['text_length'])

    # Concatenate dataframes horizontally (axis=1) to join as columns and standard it
    df = pd.concat([link_result12, phone_number_result12, content_styling_result12, count_of_pronouns12, culculate_of_length12] , axis=1)
    standardized_df = (df - df.mean()) / df.std()
    
    data1 = pd.read_excel(fileName)
    # concat original data and Standardized feature vector 
    final_data = pd.concat ([data1, standardized_df] , axis=1)
    
    # Assuming the first column is the target variable and the rest are features
    X = np.array(final_data.iloc[:, 2:]) # Features
    y = np.array(final_data.iloc[:, 1]) # Target variable

    # Configure k-Fold
    kf = KFold(n_splits=3, random_state=None, shuffle=True)
    
    # Create a Random Forest and Adaboost classifier 
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
        
    # Iterate over each split
    for train_index, test_index in kf.split(X):
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = (X[train_index], 
                                            X[test_index], 
                                            y[train_index], 
                                            y[test_index])
    
        # normalize x_train
        X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float)) # float the data
        # normalize x_test
        X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float)) # float the data
    
        # Fit the model on the training data
        random_forest.fit(X_train_norm, y_train)
        adaboost.fit(X_train_norm, y_train)

        # Make predictions on the testing set
        y_pred_forest = random_forest.predict(X_test_norm)
        y_pred_adaboost = adaboost.predict(X_test_norm)
    
        # Calculate Error Rate
        error_rate_forest = 1 - accuracy_score(y_test, y_pred_forest)
        error_rate_adaboost = 1 - accuracy_score(y_test, y_pred_adaboost)
    
        # Calculate confusion matrix
        conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)
        conf_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)
    
        # Calculate precision
        precision_forest = precision_score(y_test, y_pred_forest, pos_label='spam')  # For binary classification
        precision_adaboost = precision_score(y_test, y_pred_adaboost, pos_label='spam')  # For binary classification
    
        # Calculate recall
        recall_forest = recall_score(y_test, y_pred_forest, pos_label='spam')  # For binary classification
        recall_adaboost = recall_score(y_test, y_pred_adaboost, pos_label='spam')  # For binary classification
    
        # Calculate F1 score
        f1_forest = f1_score(y_test, y_pred_forest, pos_label='spam')  # For binary classification
        f1_adaboost = f1_score(y_test, y_pred_adaboost, pos_label='spam')  # For binary classification
    
        # Print the forest results
        print("\nRandom Forest:")
        print(f"Confusion Matrix:")
        print(conf_matrix_forest)
        print(f"Precision: {precision_forest:.3f}")
        print(f"Recall: {recall_forest:.3f}")
        print(f"F1 Score: {f1_forest:.3f}")
        print(f"Error Rate: {error_rate_forest:.3f}")
    
        # Print the adaboost results
        print("\nAdaboost:")
        print(f"Confusion Matrix:")
        print(conf_matrix_adaboost)
        print(f"Precision: {precision_adaboost:.3f}")
        print(f"Recall: {recall_adaboost:.3f}")
        print(f"F1 Score: {f1_adaboost:.3f}")
        print(f"Error Rate: {error_rate_adaboost:.3f}")
        print("\n***********************************************")    
    
    # culculate average of above values using ross_validate function
    # Create scorers
    scoring = {
       'accuracy': make_scorer(accuracy_score),
       'precision': make_scorer(precision_score, pos_label='spam'),
       'recall': make_scorer(recall_score, pos_label='spam'),
       'f1_score': make_scorer(f1_score, pos_label='spam')
    }

    # Evaluate RandomForest model
    rf_cv_results = cross_validate(random_forest, X, y, cv=kf, scoring=scoring)
    
    # Perform cross-validation and get the predicted labels
    y_pred_rf = cross_val_predict(random_forest, X, y, cv=kf)

    # Calculate confusion matrix
    cm_rf = confusion_matrix(y, y_pred_rf)
    
    print("\nAverage RandomForest:")
    #print(rf_cv_results)
    print("Average Error Rate:", 1-(np.mean(rf_cv_results['test_accuracy'])))
    print("Average Precision:", np.mean(rf_cv_results['test_precision']))
    print("Average Recall:", np.mean(rf_cv_results['test_recall']))
    print("Average F1 Score:", np.mean(rf_cv_results['test_f1_score']))
    print("Confusion Matrix:")
    print(cm_rf)

    # Evaluate AdaBoost model
    ab_cv_results = cross_validate(adaboost, X, y, cv=kf, scoring=scoring)
    
    # Perform cross-validation and get the predicted labels
    y_pred_ad = cross_val_predict(adaboost, X, y, cv=kf)

    # Calculate confusion matrix
    cm_ad = confusion_matrix(y, y_pred_ad)
    
    print("\nAverage AdaBoost:")
    #print(ab_cv_results)
    print("Average accuracy:", 1-(np.mean(ab_cv_results['test_accuracy'])))
    print("Average Precision:", np.mean(ab_cv_results['test_precision']))
    print("Average Recall:", np.mean(ab_cv_results['test_recall']))
    print("Average F1 Score:", np.mean(ab_cv_results['test_f1_score']))
    print("Confusion Matrix:")
    print(cm_ad) 
    print() 