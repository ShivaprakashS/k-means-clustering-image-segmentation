import sys
import random
import numpy as nm
from skimage import io
from skimage import img_as_ubyte
import numpy.matlib
import warnings
import imageio
#reading the image and pixels from 0 to 255 for representation.
def read_image(image1):
    image = io.imread(image1)
    image = image/255
    rows = image.shape[0]
    cols = image.shape[1]
    return image,rows,cols    

#calculating the size
def size(im):
    size=nm.size(im,0)
    return size
#calculating the zeros using numpy
def zeros(im):
    zeros=nm.zeros((size(im),1))
    return zeros
#calculating the mean choosing the clusters and centroids (kmeans)
def means(randomclust,im):
    K = size(randomclust)
    clusters = zeros(im)
    clusters = chooseclusters(randomclust,im)
    centroids = centroid(clusters,im,K)
    return clusters,centroids
#calculating the cluster array and resizing.
def clusterarrayresize(randomclust,im):
    clusterarray = nm.empty((size(im),1))
    for i in range(0,size(randomclust)):
        randomcluster = nm.sum(nm.power(nm.subtract(im,nm.ones((size(im),1))*randomclust[i]),2),axis = 1)
        randomcluster1 = nm.asarray(randomcluster)
        randomcluster1.resize((size(im),1))
        clusterarray = nm.append(clusterarray, randomcluster1, axis=1)
    return clusterarray   
#Initially choosing the cluster centroids and repeat until convergence argmin of array of clusters. 
def chooseclusters(randomclust,im):
    clusters = zeros(im)
    clusterarray=clusterarrayresize(randomclust,im)
    #removing the subarrry of the obj 0 in the cluster array
    clusterarray1 = nm.delete(clusterarray,0,axis=1)
    #returning the minimum indices clusters axis=1 for col
    clusters = nm.argmin(clusterarray1, axis=1)
    return clusters

#total number of cluster and data points assigned to those clusters
def clustertotal(clusters1,im):
    clusters1 = clusters1.astype(int)
    sum1 = sum(clusters1);
    clusters1.resize((size(im),1))
    #reshaping the matrix clusters to 1 row and columns of image size (axis=1)
    mat = nm.matlib.repmat(clusters1,1,nm.size(im,1))
    return clusters1,sum1,mat
#finding the centroid of the clusters.
def centroid(clusters,im,K):
    centroids = nm.zeros((K,nm.size(im,1)))
    for i in range(0,K):
        clusters1 = (clusters==i)
        clusters1,sum1,mat=clustertotal(clusters1,im)
        clusters1 = nm.transpose(clusters1)
        #summing the rows of image matrix 
        summat=nm.sum(nm.multiply(im,mat),axis=0)
        #centroids
        centroids[i] = summat*(1/sum1)
    return centroids


if __name__ == "__main__":
    K=int(sys.argv[1])
    image1=sys.argv[2]
    image,rows,cols=read_image(image1)
    prod=rows*cols
    randomclust = random.sample(list(image.reshape(prod,3)),K)
    clusters,centroids = means(randomclust,image.reshape(prod,3))
    #Assignment Step
    clusters = chooseclusters(centroids,image.reshape(prod,3))
    clustersimage = centroids[clusters]
    #Update Step
    clustersimage = nm.reshape(clustersimage, (rows, cols, 3))
    with warnings.catch_warnings():
     warnings.simplefilter("ignore")
     clustersimage=img_as_ubyte(clustersimage)

    imageio.imwrite('compressedimage.jpg', clustersimage)
    compressed = io.imread('compressedimage.jpg')
    io.imshow(compressed)
    io.show()
