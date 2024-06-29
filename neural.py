import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
#przygotowanie danych
def preprocess_data(data):
    df = pd.DataFrame(data[1:], columns=data[0])
    df.drop_duplicates(keep='first',inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    X = df.drop('quality', axis=1)
    y = df['quality']

    for col in df.columns:
        outliers = find_outliers_std(df, col)
        if not outliers.empty:
            df.drop(outliers.index, inplace=True)
    return X,y

def find_outliers_std(df, col):
    mean = df[col].mean()
    std_dev = df[col].std()
    outliers = df[(df[col] < mean - 3 * std_dev) | (df[col] > mean + 3 * std_dev)]
    return outliers

class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}")


class TensorNeuralNetwork():
    def __init__(self, hidden_layer_sizes=(16,), n_classes=16, learning_rate=0.001, n_iters=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.model = None

    def build_model(self, n_features):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        for units in self.hidden_layer_sizes:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_classes, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.model = self.build_model(n_features)
        print_loss_callback = PrintLossCallback()
        self.model.fit(X, y, epochs=self.n_iters, verbose=0, callbacks=[print_loss_callback])

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def metric(self,y,y_pred):
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 score: {f1:.4f}')
        return precision,recall,accuracy,f1

def build_and_train_model(X, y, k=5):
    skf = StratifiedKFold(n_splits=k)
    models = []
    conf_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler= StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
        model = TensorNeuralNetwork(hidden_layer_sizes=(16,8,6), learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision, recall, accuracy, f1 = model.metric(y_test,y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        models.append(model)
        conf_matrices.append(conf_matrix)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print("\nAverage Metrics:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    return models, conf_matrices


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('predict')
    plt.ylabel('real')
    plt.show()

def evaluate_model(model, X_train,y_train,X_test,y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    return conf_matrix, test_accuracy

def get_user_input():
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
    user_data = []
    print("insert values:")
    for col in columns:
        value = float(input(f"{col}: "))
        user_data.append(value)

    return user_data

def predict_quality(model, mean, std, user_data):
    user_data = (user_data - mean) / std
    if isinstance(model, TensorNeuralNetwork):
        user_data = np.array(user_data).reshape(1, -1)

    prediction = model.predict([user_data])
    return prediction[0]


