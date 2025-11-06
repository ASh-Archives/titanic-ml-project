import pandas as pd
from sklearn.externals import joblib

def predict(model, test_data):
    predictions = model.predict(test_data)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    submission.to_csv('submission.csv', index=False)