from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # PreProcess the data by scaling and splitting
    data.drop(['EmployeeCount', 'EmployeeNumber','Over18','StandardHours'], axis="columns", inplace = True)
    
    categorical_col = []
    for column in data.columns:
        if data[column].dtype == object and len(data[column].unique()) <= 50:
            categorical_col.append(column)

    # Here the attrition column in the data is being converted to a categorical data type using .astype("category")
    data['Attrition'] = data.Attrition.astype("category").cat.codes
    
    categorical_col.remove('Attrition')
    
    label = LabelEncoder()
    for column in categorical_col:
        data[column] = label.fit_transform(data[column])
    
    print(data.head())
    
    #We are going to drop the target variable "Attrition" leaving only input feature
    X = data.drop('Attrition', axis = 1)
    y=data.Attrition
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test