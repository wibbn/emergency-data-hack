from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import pandas as pd

def train_loop(data):
    data = data.drop(data[data['target'] == 0].sample(frac=.98).index)

    train, test = train_test_split(data, test_size=0.2)

    print("Размер выборки для обучения: ", train.shape)
    print("Размер выборки для тестирования: ", test.shape)

    columns = ['volume',
           'occupancy',
           'speed',
           'repair']
    
    train_x = train[columns]
    train_y = train['target']

    test_x = test[columns]
    test_y = test['target']

    clf = CatBoostClassifier(class_weights=[1, 2, 4])
    clf.fit(train_x, train_y)

    prediction = clf.predict(test_x)

    score = f1_score(test_y, prediction, average='macro')

    print(f"F1-score: {score}")

    print(pd.concat({
        'predicted': pd.Series(prediction.flatten()).astype(int).value_counts(), 
        'truth': test_y.value_counts()
    }, axis=1))

    return clf