import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read source image.

    im_src = cv2.imread('source.png')
    # Four corners of the book in source image
    pts_src = np.array([[334, 376], [73, 249], [371, 217], [162, 76]])
    # Read destination image.
    im_dst = cv2.imread('destination.png')
    # Four corners of the book in destination image.
    pts_dst = np.array([[274, 127], [274, 401], [439, 147], [461, 398]])
    # Calculate Homograph
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    cv2.imwrite('Source Image.png', im_src)
    cv2.imwrite('Destination Image.png', im_dst)
    cv2.imwrite('Warped Source Image.png', im_out)
    
    cv2.waitKey(0)
