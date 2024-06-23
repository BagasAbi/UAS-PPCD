import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Ensure all elements in 'data' have the same length
max_landmarks = 42  # This should match the padding length in create_dataset.py
for item in data:
    if len(item) != max_landmarks:
        raise ValueError(f"Data item length {len(item)} does not match expected length {max_landmarks}")

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
