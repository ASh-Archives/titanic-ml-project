from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Заполнение пропущенных значений
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Создание нового признака
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # Кодирование категориальных признаков
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Разделение на признаки и целевую переменную
    X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    y = df['Survived']
    
    return X, y