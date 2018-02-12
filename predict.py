'''
ESE650 Project1 by Wudao Ling
use this code to use gaussian model for color segmentation
run this after train.py
'''

import cv2, os, pickle
import numpy as np
from train import gaussian_likelihood, gmm_likelihood
from math import sqrt

def main():
    # select model
    model = pickle.load(open("gmm_model_all.p", "rb"))

    dist_weight = model['dist_weight']
    red_mean = model['red_mean']
    red_cov = model['red_cov']
    red_prob = model['red_prob']
    trick_mean = model['trick_mean']
    trick_cov = model['trick_cov']
    trick_prob = model['trick_prob']
    others_mean = model['others_mean']
    others_cov = model['others_cov']
    # test_set = model['test_set']

    # load test image
    folder = "roipoly_annotate/test"

    # test_set = []
    # for file in os.listdir(folder):
    #     filename = os.path.splitext(file)[0]
    #     test_set.append(filename)

    # for id, filename in enumerate(test_set):

    for id, file in enumerate(os.listdir(folder)):
        filename = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(folder, filename + ".png"))
        #visualize('test '+filename, img)

        # segmented image
        img_seg = seg_color(img,red_mean,red_cov,red_prob, trick_mean, trick_cov, trick_prob, others_mean,others_cov)
        #visualize('seg '+filename, img_seg)

        # barrel bounding box + distance of barrel
        img_box, centroids, distances = detect_barrel(img,img_seg,dist_weight)
        visualize('box '+filename+', estimated: '+str(distances[:]), img_box)

        if len(distances)==0:
            print('ImageNo = [' + str(id + 1) + '], find no red barrel.')
        else:
            print('ImageNo = [' + str(id + 1) + '], CentroidX = ' + str(centroids[:, 0]) +
                  ', CentroidY = ' + str(centroids[:, 1]) + ', Distance = ' + str(distances[:]) +
                  ', found '+str(len(distances))+' red barrel. ')

def visualize(title,img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seg_color(img,focus_mean,focus_cov,focus_prob, trick_mean, trick_cov, trick_prob, others_mean,others_cov):
    # reshape img to (row*col,3), record reshape first
    shape = img.shape[:2]
    img_seg = np.zeros(shape, dtype=np.uint8)
    pixels = np.reshape(img, (-1, 3))

    if focus_mean.ndim == 1 and trick_mean.ndim == 1 and others_mean.ndim == 1:
        # unimodel gaussian
        focus_likelihood = gaussian_likelihood(pixels,focus_mean,focus_cov)
        trick_likelihood = gaussian_likelihood(pixels, trick_mean, trick_cov)
        others_likelihood = gaussian_likelihood(pixels, others_mean, others_cov)
    else:
        # GMM
        focus_likelihood = gmm_likelihood(pixels,focus_mean,focus_cov)

        trick_likelihood = gmm_likelihood(pixels,trick_mean,trick_cov)

        if others_mean.ndim == 1: #hybrid GMM
            others_likelihood = gaussian_likelihood(pixels, others_mean, others_cov)
        else:
            others_likelihood = gmm_likelihood(pixels,others_mean,others_cov)

    focus_posterior = focus_likelihood*focus_prob
    others_posterior = others_likelihood*(1 - focus_prob)
    trick_posterior = trick_likelihood*trick_prob*3

    mask = ~np.ma.mask_or(focus_posterior<others_posterior,focus_posterior<trick_posterior)
    mask = np.reshape(mask,shape)
    img_seg[mask] = 255

    # post-process
    # remove noises (erosion followed by dilation)
    kernel = np.ones((15, 15), np.uint8) #20,15
    img_seg = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    return img_seg

def detect_barrel(img, img_focus, dist_weight):
    # init
    centroids = []
    distances = []

    # use openCV contour
    ret, thresh = cv2.threshold(img_focus, 127, 255, 0)
    img_cnt, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours)==0:
        return img,centroids,distances

    max_area = np.max([cv2.contourArea(contour) for contour in contours])

    # detect barrel
    for contour in contours:
        # check area, abandon relatively/absolutely small area
        area = cv2.contourArea(contour)
        if area < 700:
            continue
        if area < max_area/5: #10
            continue

        # rotated bounding box, for debug
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        # build bounding box, abandon wrong shape
        x, y, w, h = cv2.boundingRect(contour)
        if h/w<1.15 or h/w>2.6:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (round(x+w/2),round(y+h/2)), 5, (0,255,0), -1, 8)

        centroids.append((x+w/2,y+h/2))
        distances.append((dist_weight/sqrt(w*h)).item())

    return img, np.array(centroids), np.array(distances)

if __name__ == '__main__':
    main()