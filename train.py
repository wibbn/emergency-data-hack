from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import pandas as pd

def train_loop(data, cats, class_weights=[1, 1, 1]):
    data = data.drop(data[data['target'] == 0].sample(frac=.98).index)

    train, test = train_test_split(data, test_size=0.2)

    print("Размер выборки для обучения: ", train.shape)
    print("Размер выборки для тестирования: ", test.shape)
    
    train_x = train.drop('target', axis=1)
    train_y = train['target']

    test_x = test.drop('target', axis=1)
    test_y = test['target']

    train_pool = Pool(train_x, train_y, cat_features=cats)
    test_pool = Pool(test_x, test_y, cat_features=cats)

    clf = CatBoostClassifier(class_weights=class_weights, logging_level='Silent')
    clf.fit(train_pool)

    prediction = clf.predict(test_pool)

    score = f1_score(test_y, prediction, average='macro')

    print(f"F1-score: {score}")

    print(pd.concat({
        'predicted': pd.Series(prediction.flatten()).astype(int).value_counts(), 
        'truth': test_y.value_counts()
    }, axis=1))

    return clf, score