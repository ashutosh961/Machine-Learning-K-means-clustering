from __future__ import division
import sys
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np

#GREEN POINTS ARE IRIS VIRGINICA
#RED POINTS ARE IRIS SETOSA
#BLUE POINTS ARE IRIS VERSICOLOR
#MATPLOTLIB LEGEND ISSUE SO CANNOT SHOW LEGEND

#ASHUTOSH UPADHYE

def load_data():
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    #print(labels)
    return dict(zip(features, labels))

def distance_euclid(f1, f2):
    a_vector = np.array
    d_vector = a_vector(f1)-a_vector(f2)
    return np.sqrt(np.dot(d_vector, d_vector))

def mean(vectors):
    return tuple(np.mean(vectors, axis=0))

def assign(centers):
    new_centers = defaultdict(list)
    for cx in centers:
        for x in centers[cx]:
            best = min(centers, key=lambda c: distance_euclid(x, c))
            new_centers[best] += [x]
    return new_centers

def update(centers):
    new_centers = {}
    for c in centers:
        new_centers[mean(centers[c])] = centers[c]
    return new_centers

def kmeans(features, k, maxiter=100):

    centers = dict((c, [c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in range(maxiter):
        new_centers = assign(centers)
        new_centers = update(new_centers)
        #print(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
    return centers

def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)

def Sort_into_clusters(k_clusters,seed=123):
    try:
        data = load_data()
    except IOError:
        print("Missing dataset! Run:")
        sys.exit(1)
    features = data.keys()
    #print("features:"+str(features))
    random.seed(seed)
    random.shuffle(list(features))
    clusters = kmeans(list(features), k_clusters)
    #print(clusters.values())
    labels = []
    data_values = []
    centroids = []
    sepal_list =[]
    petal_list = []
    colors = []

    for keys in clusters.keys():
        centroids.append(keys)


    for c in clusters:
        print(counter([data[x] for x in clusters[c]]))
        for x in clusters[c]:
            print([x])
            data_values.append(x)
            if(data[x]=='Iris-setosa'):
                labels.append('Iris-setosa')
                colors.append('r')
            if (data[x] == 'Iris-virginica'):
                labels.append('Iris-virginica')
                colors.append('g')
            if (data[x] == 'Iris-versicolor'):
                labels.append('Iris-versicolor')
                colors.append('b')
    print(colors)
    print(labels)
    i =0
    for data_points in data_values:
        sepal_list.append([data_points[0]])
        petal_list.append([data_points[1]])
        plt.scatter(data_points[0], data_points[1], c=colors[i:])
        if i < len(colors)-1:i += 1
    for centroid_points in centroids:
        plt.scatter(centroid_points[0], centroid_points[1], marker='^', s=170, zorder=20, c='r')
    #plt.plot(sepal_list,petal_list, color=list(labels))
    plt.xlabel("Sepal width")
    plt.ylabel("Petal length")
    print(np.unique(labels))
    plt.show()

if __name__ == "__main__":
    k_clusters = input("Please enter k for k-means:")
    print(k_clusters)
    Sort_into_clusters(int(k_clusters))