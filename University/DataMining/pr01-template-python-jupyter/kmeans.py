
# K-means algorithm by Obada Ghazlan

import numpy as np
import cv2
from numpy.random import seed
import random
import math
from matplotlib import pyplot as plt, gridspec, cm
from PIL import Image
from scipy.spatial import distance
import statistics


def get_feature_vector(url):
    # read the image
    img = cv2.imread(url)
    # convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixels = img.reshape(-1, 3)
    # convert to float
    pixels = np.float32(pixels)
    # print(len(pixels))
    return pixels
    # for i in range(len(pixels)):
    # print(pixels[i])


def random_centroids(k, centroids, vector):
    # generate k random 3 dimensional vector from the the data
    # and append it to the centroid array those points will be considered as the initial "clusters"
    random_cent = random.sample(range(0, (len(vector) - 1)), k)
    for i in range(k):
        centroids.append(vector[random_cent[i]])

    return centroids


def assign_points(centroids, points, k):
    # generate the centroids
    cent = random_centroids(k, centroids, points)
    cost = 0
    dst = []
    label = []
    # calculate the euclidean distance between each point and the centroids
    # check which centroid is the closest then create a label array that saves the number of the cluster
    for i in range(len(points)):
        for j in range(k):
            dst.append(distance.euclidean(points[i], cent[j]))

        # calcuate the minimum distance of distance array
        minimum_distance = min(dst)
        cost += minimum_distance
        ind = dst.index(minimum_distance)
        label.append(ind)
        dst.clear()
    # print(label)
    return label


# help function to calculate the mean of vectors
def recalculate_mean(cluster):
    new_mean = np.mean(cluster, axis=0)
    return new_mean


def recalculate_clusters(labels, centroids, points):
    tmp = []
    clusters = []
    # for each cluster recalculate the mean
    for i in range(len(centroids)):
        for j in range(len(labels)):
            if i == labels[j]:
                tmp.append(points[j])

        new_mean = recalculate_mean(tmp)
        clusters.append(new_mean)

    return clusters


if __name__ == "__main__":
    im = input("Enter the image url: ")
    image = cv2.imread(im)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    k1 = 6
    get_points = get_feature_vector(im)
    centroid = random_centroids(k1, [], get_points)
    iteration = 0
    prev_cost = 1
    label1 = assign_points(centroid, get_points, k1)
    while iteration <= 50:
        print('ITERATION ' + str(iteration))
        kutta = recalculate_clusters(label1, centroid, get_points)
        assign_points(kutta, get_points, k1)
        iteration += 1

    centers = np.uint8(centroid)
    labels = np.uint8(label1)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]

    segmented_image = segmented_image.reshape(image.shape)

    plt.imshow(segmented_image)
    plt.show()