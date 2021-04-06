import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# May not work well for skewed distributions
def zscore_standardization(data):
    for i, column_data in enumerate(data.transpose()):
        data[:, i] = (column_data - column_data.mean()) / column_data.std()
    return data

def accuracy(predictions, labels):
    correct_predictions = 0
    n_predictions = len(predictions)

    for index in range(len(predictions)):
        if predictions[index] == labels[index]:
            correct_predictions += 1
    return correct_predictions / n_predictions
    
def get_predictions(train_data,
                    test_data,
                    labels, k):
    similarities = euclidean_distances(test_data, train_data)
    predictions = []
    for i, similarity in enumerate(similarities):
        # For each point choose top k neighbors
        k_prediction_indices = np.argsort(similarity)[:k]
        temp_list = [labels[i] for i in k_prediction_indices]
        predictions.append(max(temp_list, key=temp_list.count))

    return predictions
