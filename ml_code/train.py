from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Train and evaluate the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    if hasattr(model, "predict_proba"):
        # predict_proba return a matrix of the two columns (negative and positive), [:,1] means probability
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        # if model doesn't have
        y_prob = model.decision_function(X_test)
        
    return accuracy, cm, y_test, y_prob