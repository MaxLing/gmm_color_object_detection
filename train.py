'''
ESE650 Project1 by Wudao Ling
use this code to train gaussian model for color segmentation
run this after annotate.py (hand-labeling)
'''

import os, cv2, pickle
import numpy as np
from math import pi, sqrt
from sklearn.linear_model import LinearRegression

def load_data():
    # init a list of pixels for easier append
    red = []
    trick = []
    others = []
    distance = []
    reciprocal_sqrt_area = []

    # test/training split
    #id_test = np.random.choice(50, 10, replace=False)]
    id_test=[]
    test_set = []

    folder_red = "roipoly_annotate/labeled_data/RedBarrel"
    folder_trick = "roipoly_annotate/labeled_data/TrickyCases"
    folder_image = "roipoly_annotate/train"
    file_list = os.listdir(folder_red)
    for id, file in enumerate(file_list):
        filename = os.path.splitext(file)[0]

        # skip and record test set
        if id in id_test:
            test_set.append(filename)
            continue

        # read data: height*width binary mask
        mask_red = np.load(os.path.join(folder_red, filename + ".npy"))
        mask_trick = np.load(os.path.join(folder_trick, filename + ".npy"))
        # read img
        img = cv2.imread(os.path.join(folder_image, filename + ".png"))

        # save pixels n*3
        pixel_red = img[mask_red]
        pixel_trick = img[mask_trick]
        pixel_others = img[~mask_red]
        red.extend(pixel_red)
        trick.extend(pixel_trick)
        others.extend(pixel_others)

        # save distances and area for regression
        dist = filename.split('.')[0]
        if dist.count('_') != 0:
            continue # abandon cases with multiple barrel
        distance.append(int(dist))
        area = sum(sum(mask_red))
        reciprocal_sqrt_area.append(1/sqrt(area))


    # convert to array
    return np.array(red), np.array(trick), np.array(others), test_set, np.array(distance).reshape(-1,1), np.array(reciprocal_sqrt_area).reshape(-1,1)

def gaussian_likelihood(data, mean, cov):
    L = np.linalg.cholesky(np.linalg.inv(cov))
    exp_term = np.exp(-0.5*np.sum(np.square(np.dot(data-mean,L)),axis = 1))
    likelihood = 1/(np.sqrt(((2*pi)**3)*(np.linalg.det(cov))))*exp_term
    return likelihood

def gmm_likelihood(data, mean, cov):
    K, m = mean.shape
    likelihood = 0
    for i in range(K):
        likelihood += gaussian_likelihood(data, mean[i], cov[i] * np.eye(m))
    return likelihood/K

def gaussian_md(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    return mean,cov

def gaussian_mixture_model(data, K):
    # EM algorithm (diagonal covariance)
    # init
    n, m = data.shape
    #split = np.array_split(data, K)
    #mean = np.array([np.mean(s, axis=0) for s in split]) # randomly sample K pixels as mean
    mean = np.random.uniform(low=0, high=1, size = (K, m))*255.
    cov = np.ones((K,m)) * 1000
    meb_prob = np.ones((n, K))/K  # equal at the beginning

    thresh = 1

    for i in range(50):
        # record last mean for convergence check, must be softcopy
        last_mean = np.copy(mean)

        # E step
        for j in range(K):
            # element-wise
            meb_prob[:,j] = gaussian_likelihood(data,mean[j],cov[j]*np.eye(m))
        # normalization
        meb_prob = np.divide(meb_prob,sum(meb_prob.T).reshape(n,1))

        # M step
        for j in range(K):
            meb_prob_reshaped = meb_prob[:,j].reshape((n,1))
            mean[j] = np.dot(meb_prob_reshaped.T,data)/sum(meb_prob_reshaped)
            cov[j] = np.dot(meb_prob_reshaped.T,(data-mean[j])**2) / sum(meb_prob_reshaped) # diagonal covariance

        print('EM iteration '+str(i+1))
        print(mean)
        print(cov)

        # check mean vector converged or not
        if np.linalg.norm(mean-last_mean)<thresh:
            break

    return mean,cov

def distance_regression(reciprocal_sqrt_area, distance):
    estimator = LinearRegression(fit_intercept=False)
    estimator.fit(reciprocal_sqrt_area, distance)
    return estimator.coef_

def main():
    red, trick, others, test_set, distance, reciprocal_sqrt_area = load_data()

    dist_weight = distance_regression(reciprocal_sqrt_area, distance)

    red_prob = len(red)/(len(red)+len(others))
    trick_prob = len(trick)/(len(red)+len(others))
    red_mean, red_cov = gaussian_mixture_model(red,3) #3
    trick_mean, trick_cov = gaussian_mixture_model(trick,10) #10
    others_mean, others_cov = gaussian_md(others)

    gmm = {'red_mean': red_mean, 'red_cov': red_cov, 'red_prob': red_prob,
           'trick_mean': trick_mean, 'trick_cov': trick_cov, 'trick_prob': trick_prob,
           'others_mean': others_mean, 'others_cov': others_cov,
           'test_set': test_set, 'dist_weight':dist_weight}
    pickle.dump(gmm, open("gmm_model_test.p", "wb"))

if __name__ == '__main__':
    main()