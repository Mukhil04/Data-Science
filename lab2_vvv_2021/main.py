import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        inverse = np.linalg.inv(cov)
        term1 = np.log(pi)
        labels = np.argmax((term1 + np.dot(np.dot(means,inverse), data.T).T - 0.5*np.sum(np.dot(means,inverse)*means, axis=1)), axis = 1)
        return labels

    def classifierError(self,truelabels,estimatedlabels):
        sum = 0
        sample = np.equal(truelabels,estimatedlabels)
        for i in sample:
            if i:
                pass
            else:
                sum = sum + 1
        error = sum/truelabels.size
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # Put your code below
        arr0 = trainfeat[trainlabel==0]
        arr1 = trainfeat[trainlabel==1]
        arr2 = trainfeat[trainlabel==2]

        for i in range(nlabels):
            pi[i] = trainfeat[trainlabel==i].size/(2*trainfeat.shape[0])

        for i in range(int(arr0.size/2)):
            means[0] = means[0] + arr0[i]
        means[0] = means[0]/int(arr0.size/2)
        for i in range(int(arr1.size/2)):
            means[1] = means[1] + arr1[i]
        means[1] = means[1]/int(arr1.size/2)
        for i in range(int(arr2.size/2)):
            means[2] = means[2] + arr2[i]
        means[2] = means[2]/int(arr2.size/2)


        for a in arr0:
            cov = cov + np.outer((a-means[0]),(a-means[0]))
        for a in arr1:  
            cov = cov + np.outer((a-means[1]),(a-means[1]))
        for a in arr2:
            cov = cov + np.outer((a-means[2]),(a-means[2]))
        cov = cov/(trainfeat.shape[0] - nlabels)
        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi, means, cov = self.trainLDA(trainingdata,traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata, pi, means, cov)
        trerror = q1.classifierError(traininglabels, esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi, means, cov = self.trainLDA(trainingdata, traininglabels)
        estvallabels = q1.bayesClassifier(valdata, pi, means, cov)
        valerror = q1.classifierError(vallabels, estvallabels)
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        labels = np.zeros(testfeat.shape[0])
        arr = trainlabel[(np.argpartition(dist.cdist(testfeat, trainfeat),k))]
        for i in range(arr.shape[0]):
            m = stats.mode(arr[i][:k], axis = None)
            labels[i] = m[0][0]
        return labels

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]
        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            labels1 = self.kNN(trainingdata, traininglabels, trainingdata,k_array[i])
            trainingError[i] = q1.classifierError(traininglabels,labels1)
            labels2 = self.kNN(valdata, vallabels, valdata, k_array[i])
            validationError[i] = q1.classifierError(vallabels,labels2)
        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        neigh = neighbors.KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', p = 2)
        t0 = time.clock()
        neigh.fit(traindata, trainlabels)
        t1 = time.clock() - t0
        t2 = time.clock()
        neigh.predict(valdata)
        t3 = time.clock() - t2 
        classifier, valerror, fitTime, predTime = (neigh, 1 - neigh.score(valdata, vallabels), t1, t3)
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        a = LinearDiscriminantAnalysis()
        t0 = time.clock()
        a.fit(traindata, trainlabels)
        t1 = time.clock() - t0
        t2 = time.clock()
        a.predict(valdata)
        t3 = time.clock() - t2 
        classifier, valerror, fitTime, predTime = (a, 1 - a.score(valdata, vallabels), t1, t3)

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
