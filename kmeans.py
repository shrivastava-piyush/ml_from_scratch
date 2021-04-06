import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class KMeans:

  def __init__(self,
               k,
               epochs,
               data):
    self.k = k
    self.epochs = epochs

    self.data = data
    #Initialize centroids randomly based on data shape
    centroid_indexes = np.random.choice(self.data.shape[0], k, replace=True)
    self.centroids = self.data[centroid_indexes, :]

  def calculate_distances(self,
                          data):
    distances = euclidean_distances(self.centroids, data)
    return distances

  def train(self):
    for i in range(self.epochs):
      if i % 10 == 0:
        print("Epoch {}...".format(i))

      prev_centroids = np.copy(self.centroids)

      distances = self.calculate_distances(self.data)
      # Choose labels based on minimum distance from a centroid
      labels = np.argmin(distances, axis=0)

      for n in range(self.k):
        self.centroids[n] = np.mean(self.data[labels == n, :], axis=0)

      if np.array_equal(prev_centroids, self.centroids):
        print("Convergence reached")
        break

  def predict(self):
    distances = self.calculate_distances(self.data)
    return np.argmin(distances, axis=0)

  def accuracy(self,
               predictions, 
               labels):
    return (predictions == labels).mean()

  def purity(self,
           predictions,
           labels):
    bins = np.concatenate((np.unique(labels), [np.max(np.unique(labels)) + 1]), axis=0)
    counts = [np.histogram(labels[predictions == pred], bins=bins)[0] for pred in np.unique(predictions)]

    true_label_total = np.sum([np.max(count) for count in counts])
    return true_label_total/predictions.shape[0]

  def gini_index(self,
                 predictions,
                 labels):
    bins = np.concatenate((np.unique(labels), [np.max(np.unique(labels)) + 1]), axis=0)
    counts = [np.histogram(labels[predictions == pred], bins=bins)[0] for pred in np.unique(predictions)]

    gini_density = 0
    gini_num = 0

    for count in counts:
      _sum = 0
      c_total = np.sum(count)
      for c in count:
        _sum += pow(c/c_total, 2)
      gini_density += c_total
      gini_num += (1- _sum) * c_total

    return gini_num/gini_density
