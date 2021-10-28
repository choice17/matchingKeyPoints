# https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/
# https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html

import cv2 
import numpy as np
import matplotlib.pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.3

def main():
    im1=cv2.imread("img1.jpg")
    im2=cv2.imread("img2.jpg")
    H, W, C = im1.shape
    pts = [0,0,W,0,W,H,0,H]
    pts = np.array(pts).reshape(-1, 1, 2).astype(np.float32)
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    '''
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    outFilename = "aligned.jpg"
    cv2.imwrite(outFilename, im1Reg)
    '''
    

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = im2.shape
    ptsReg = cv2.perspectiveTransform(pts, h)

    #ptsReg = np.array([0, 0, 100, 0, 100, 100, 100, 0]).reshape(-1,1,2)
    cv2.polylines(im2,[np.int32(ptsReg)],True,(0,255,0),3)
    outFilename = "matcing.jpg"
    cv2.imwrite(outFilename, im2)

    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    outFilename = "aligned.jpg"
    cv2.imwrite(outFilename, im1Reg)

    


def main_example():

    #read images
    img_1c=cv2.imread("img1.jpg")
    img_2c=cv2.imread("img2.jpg")

    #transform images into gray scale
    img1 = cv2.cvtColor(img_1c, cv2.COLOR_BGR2GRAY )
    img2 = cv2.cvtColor(img_2c, cv2.COLOR_BGR2GRAY )
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    
    plt.imshow(img3)
    plt.show()
    

main()
