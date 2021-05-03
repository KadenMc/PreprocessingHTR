import numpy as np
import cv2
import math


def get_equidistant_circle_points(r, num_points=8):
    """Gets equidistant points on a circle."""
    points = []
    for index in range(num_points):
        points.append([r*math.cos((index*2*math.pi)/num_points),
                      r*math.sin((index*2*math.pi)/num_points)])
    return points


def page_hole_removal(img):
    """Removes page ring holes, should they exist.
    
    Parameters:
        img (np.ndarray): The image for which to remove page holes.
    """
    detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = (img.shape[0]*img.shape[1])*0.0005
    params.maxArea = (img.shape[0]*img.shape[1])*0.003

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.05

    # Filter by Convexity
    params.filterByConvexity = False
    # params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Distance Between Blobs
    params.minDistBetweenBlobs = img.shape[0]//8

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(img)

    # Number of points around circle to sample color
    num_points = 20

    # Used when we try to sample a color off-screen
    adopt_next = False

    # Iterate over the circles
    for k in keypoints:

        # Calculate the average color around the circle
        color_points = get_equidistant_circle_points(
            (k.size/2)*1.15, num_points=num_points)
        colors = np.zeros((num_points, 3))
        for c, p in enumerate(color_points):
            try:
                colors[c] = img[int(
                    k.pt[1]) + int(p[1]), int(k.pt[0]) + int(p[0])]

                if adopt_next:
                    i = 1
                    while np.all((colors[c - i] == 0)):
                        colors[c - i] = colors[c]
                        i -= 1
                    adopt_next = False

            except:
                # Adopt previous (if the previous isn't also nothing)
                if c > 0 and not np.all((colors[c - 1] == 0)):
                    colors[c] = colors[c - 1]
                # Signal to adopt the next color
                else:
                    adopt_next = True
        
        color = colors.mean(axis=0).astype(int)
        color = (int(color[0]), int(color[1]), int(color[2]))

        # Fill the circle with the average color around it
        cv2.circle(img, (int(k.pt[0]), int(
            k.pt[1])), int((k.size/2)*1.15), color, -1)

        # Blur the area around the circle to avoid it being picked up by canny
        left = int(k.pt[0]) - int((k.size/2)*1.2)
        right = int(k.pt[0]) + int((k.size/2)*1.2)
        top = int(k.pt[1]) - int((k.size/2)*1.2)
        bottom = int(k.pt[1]) + int((k.size/2)*1.2)
        img_to_blur = img[top:bottom, left:right]
        blurred = cv2.GaussianBlur(img_to_blur, (9, 9), 0)
        img[top:bottom, left:right] = blurred
    
    return img