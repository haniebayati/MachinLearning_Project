# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import preprocessing
import jdatetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
    
# read excel files    
data1 = pd.read_excel(r'AED-USD-Jan_2015_Dec_2022.xlsx')
data2 = pd.read_excel(r'Crude_Oil_Price-Jan_2015_Dec_2022.xlsx')
data3 = pd.read_excel(r'EUR-USD-Jan_2015_Dec_2022.xlsx')
data4 = pd.read_excel(r'Gold_Price-Jan_2015_Dec_2022.xlsx')
data5 = pd.read_excel(r'Stock_Index_13940101_14010801.xlsx')
data6 = pd.read_excel(r'USD-IRR-Nov-2011-December-2022.xlsx')

# define a methode to predict next day price
def predict_next_day():
    
    # use the price of 'today' to predict the price of 'next day'.
    data_column = data6['last price recorded on the day'] 
    X = data_column.iloc[:-1].values.reshape(-1, 1)        # features (all day except last day)
    y = data_column.iloc[1:].values.reshape(-1, 1)         # targets (all day except first day)

    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # standardize x_train and x_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train_scaled, y_train)

    # Predict the next day's price
    #y_pred = model.predict(X_test_scaled)

    # Predicting the price for the next day after the last available price in the data
    # Transform the last known price using the scaler fitted on the training data   
    # Reshape the row to have one feature and as many samples as rows available
    next_day_feature = scaler.transform(data_column.iloc[-1].reshape(-1, 1))
    # Predict the next day's price and print it
    next_day_price = model.predict(next_day_feature)
    print(f"Predicted price for the next day: {next_day_price[0]}")


# change solar date to gregorian date in excel 'Stock_Index_13940101_14010801'
data = []
for row in data5['dateissue']:
    # set year/month/day in each date
    if (str(row)).isdigit() and len(str(row)) == 8:
        year = int(str(row)[:4])
        month = int(str(row)[4:6])
        day = int(str(row)[6:])
        # change solar date to gregorian date
        gregorian_date = jdatetime.date(year, month, day).togregorian()
        data.append(gregorian_date.strftime('%Y-%m-%d'))
        
# Ensure the date columns in all files are the same data type (datetime)
data1['Date'] = pd.to_datetime(data1['Date'])
data2['Date'] = pd.to_datetime(data2['Date'])
data3['Date'] = pd.to_datetime(data3['Date'])
data4['Date'] = pd.to_datetime(data4['Date'])
data5['Date'] = pd.to_datetime(data)
data6['Date'] = pd.to_datetime(data6['gregorian calendar dates'])

# Merge datasets one by one
# If you want to merge all entries of dataset_a with dataset_b where dates match,
# but keep all entries of dataset_a even if there's no matching date in dataset_b
# You would use a 'left' join instead
merged_df = data6
merged_df = pd.merge(merged_df, data1, on='Date', how='left', suffixes=('', '_df2'))
merged_df = pd.merge(merged_df, data2, on='Date', how='left', suffixes=('', '_df3'))
merged_df = pd.merge(merged_df, data3, on='Date', how='left', suffixes=('', '_df4'))
merged_df = pd.merge(merged_df, data4, on='Date', how='left', suffixes=('', '_df5'))
merged_df = pd.merge(merged_df, data5, on='Date', how='left', suffixes=('', '_df6'))

# select required columns
selected_columns = merged_df[['Date', 'Open', 'Volume_df3', 'Open_df4', 'Volume_df5', 'Value', 'last price recorded on the day', 'highest price recorded during the day']]

# empty the cells that are (-)
selected_columns['Volume_df3'] = selected_columns['Volume_df3'].replace('-', pd.NA)
selected_columns['Volume_df5'] = selected_columns['Volume_df5'].replace('-', pd.NA)    

# Fill in the missing data in each column with the mean of that column
open_filled = selected_columns['Open'].fillna(selected_columns['Open'].mean())  
selected_columns['data-1'] = open_filled
opendf3_filled = selected_columns['Volume_df3'].fillna(selected_columns['Volume_df3'].mean())
selected_columns['data-2'] = opendf3_filled
opendf4_filled = selected_columns['Open_df4'].fillna(selected_columns['Open_df4'].mean())
selected_columns['data-3'] = opendf4_filled
opendf5_filled = selected_columns['Volume_df5'].fillna(selected_columns['Volume_df5'].mean())
selected_columns['data-4'] = opendf5_filled
value_filled = selected_columns['Value'].fillna(selected_columns['Value'].mean())
selected_columns['data-5'] = value_filled

# Calculate the difference in days between current row and the next row
selected_columns['Gap'] = selected_columns['Date'].diff(-1).dt.days.abs()
# give 0 to the last value of the column
selected_columns['Gap'].values[-1] = 0

# Calculate the percentage change to the next day
selected_columns['Pct_Change'] = selected_columns['last price recorded on the day'].pct_change(-1) * -100  # -1 because we're comparing to the next day

# Define a threshold, e.g., if the percentage change is more than 1% then it's considered a significant change
threshold = 1

# Function to classify the change
def classify_change(change, threshold):
    if change > threshold:
        return +1
    elif change < -threshold:
        return -1
    else:
        return 0

# Apply the classification to each row
selected_columns['Class'] = selected_columns['Pct_Change'].apply(classify_change, args=(threshold,))

# Remove the 'Pct_Change' from the DataFrame
selected_columns = selected_columns.drop('Pct_Change', axis=1)

# Assuming the 14th column is the target variable and the rest are features
X = selected_columns.iloc[:, 6:14] # Features
y = selected_columns.iloc[:, 14] # Target variable

# Split the data into training and testing sets (using 80%-20% split in this example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize x_train
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float)) # float the data
# normalize x_test
X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float)) # float the data

# Initialize the Support Vector and Decision Tree  and Naive Bayes and KNN Classifier
svm_Classifier = SVC(kernel='linear')  # You can choose other kernels such as 'rbf', 'poly'
dt_Classifier = DecisionTreeClassifier()
nb_Classifier = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Fit the model on the training data
svm_Classifier.fit(X_train_norm, y_train)
dt_Classifier.fit(X_train_norm, y_train)
nb_Classifier.fit(X_train_norm, y_train)
knn_model.fit(X_train_norm, y_train)

# Make predictions on train and test data
y_pred_test_svm = svm_Classifier.predict(X_test_norm)
y_pred_test_dt = dt_Classifier.predict(X_test_norm)
y_pred_test_nb = nb_Classifier.predict(X_test_norm)
y_pred_test_knn = knn_model.predict(X_test_norm)
y_pred_train_svm = svm_Classifier.predict(X_train_norm)
y_pred_train_dt = dt_Classifier.predict(X_train_norm)
y_pred_train_nb = nb_Classifier.predict(X_train_norm)
y_pred_train_knn = knn_model.predict(X_train_norm)

# Calculate Accuracy
accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
accuracy_test_knn = accuracy_score(y_test, y_pred_test_knn)
accuracy_train_svm = accuracy_score(y_train, y_pred_train_svm)
accuracy_train_dt = accuracy_score(y_train, y_pred_train_dt)
accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
accuracy_train_knn = accuracy_score(y_train, y_pred_train_knn)

# Calculate precision
precision_test_svm = precision_score(y_test, y_pred_test_svm, average='weighted')
precision_test_dt = precision_score(y_test, y_pred_test_dt, average='weighted')
precision_test_nb = precision_score(y_test, y_pred_test_nb, average='weighted')
precision_test_knn = precision_score(y_test, y_pred_test_knn, average='weighted')
precision_train_svm = precision_score(y_train, y_pred_train_svm, average='weighted')
precision_train_dt = precision_score(y_train, y_pred_train_dt, average='weighted')
precision_train_nb = precision_score(y_train, y_pred_train_nb, average='weighted')
precision_train_knn = precision_score(y_train, y_pred_train_knn, average='weighted')
    
# Calculate recall
recall_test_svm = recall_score(y_test, y_pred_test_svm, average='weighted')
recall_test_dt = recall_score(y_test, y_pred_test_dt, average='weighted')
recall_test_nb = recall_score(y_test, y_pred_test_nb, average='weighted')
recall_test_knn = recall_score(y_test, y_pred_test_knn, average='weighted')
recall_train_svm = recall_score(y_train, y_pred_train_svm, average='weighted')
recall_train_dt = recall_score(y_train, y_pred_train_dt, average='weighted')
recall_train_nb = recall_score(y_train, y_pred_train_nb, average='weighted')
recall_train_knn = recall_score(y_train, y_pred_train_knn, average='weighted')
    
# Calculate F1 score
f1_test_svm = f1_score(y_test, y_pred_test_svm, average='weighted')
f1_test_dt = f1_score(y_test, y_pred_test_dt, average='weighted')
f1_test_nb = f1_score(y_test, y_pred_test_nb, average='weighted')
f1_test_knn = f1_score(y_test, y_pred_test_knn, average='weighted')
f1_train_svm = f1_score(y_train, y_pred_train_svm, average='weighted')
f1_train_dt = f1_score(y_train, y_pred_train_dt, average='weighted')
f1_train_nb = f1_score(y_train, y_pred_train_nb, average='weighted')
f1_train_knn = f1_score(y_train, y_pred_train_knn, average='weighted')

# Print the svm results
print("\nSVM:")
print(f"\nTrain Data Precision: {precision_train_svm:.4f}")
print(f"Train Data Recall: {recall_train_svm:.4f}")
print(f"Train Data F1 Score: {f1_train_svm:.4f}")
print(f"Train Data Accuracy: {accuracy_train_svm:.4f}")

print(f"\nTest Data Precision: {precision_test_svm:.4f}")
print(f"Test Data Recall: {recall_test_svm:.4f}")
print(f"Test Data F1 Score: {f1_test_svm:.4f}")
print(f"Test Data Accuracy: {accuracy_test_svm:.4f}")

# Print the Decision Tree results
print("\nDecision Tree:")
print(f"\nTrain Data Precision: {precision_train_dt:.4f}")
print(f"Train Data Recall: {recall_train_dt:.4f}")
print(f"Train Data F1 Score: {f1_train_dt:.4f}")
print(f"Train Data Accuracy: {accuracy_train_dt:.4f}")

print(f"\nTest Data Precision: {precision_test_dt:.4f}")
print(f"Test Data Recall: {recall_test_dt:.4f}")
print(f"Test Data F1 Score: {f1_test_dt:.4f}")
print(f"Test Data Accuracy: {accuracy_test_dt:.4f}")

# Print the naive Bayes results
print("\nNaive Bayes:")
print(f"\nTrain Data Precision: {precision_train_nb:.4f}")
print(f"Train Data Recall: {recall_train_nb:.4f}")
print(f"Train Data F1 Score: {f1_train_nb:.4f}")
print(f"Train Data Accuracy: {accuracy_train_nb:.4f}")

print(f"\nTest Data Precision: {precision_test_nb:.4f}")
print(f"Test Data Recall: {recall_test_nb:.4f}")
print(f"Test Data F1 Score: {f1_test_nb:.4f}")
print(f"Test Data Accuracy: {accuracy_test_nb:.4f}")

# Print the KNN results
print("\nKNN:")
print(f"\nTrain Data Precision: {precision_train_knn:.4f}")
print(f"Train Data Recall: {recall_train_knn:.4f}")
print(f"Train Data F1 Score: {f1_train_knn:.4f}")
print(f"Train Data Accuracy: {accuracy_train_knn:.4f}")

print(f"\nTest Data Precision: {precision_test_knn:.4f}")
print(f"Test Data Recall: {recall_test_knn:.4f}")
print(f"Test Data F1 Score: {f1_test_knn:.4f}")
print(f"Test Data Accuracy: {accuracy_test_knn:.4f}")
print()

# predict next day price
predict_next_day()
print()