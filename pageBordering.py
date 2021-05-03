import numpy as np
import cv2


def page_border(img):
    """
    Crops an image of a page to the borders of said page.
    Credit: https://stackoverflow.com/questions/60145395/crop-exact-document-paper-from-image-by-removing-black-border-from-photos-in-jav

    Parameters:
        img (np.ndarray): The image to border
    """
    # Blur the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours and sort for largest contour
    cnts = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    error = True
    use_cnt = None
    display_cnt = None
    no_border = False

    # Sorted by area, find the largest contour which is approximately rectangular
    # Crop to this largest approximate rectangle
    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check if a rectangle
        if len(approx) == 4:
            error = False
            display_cnt = approx.reshape(4, 2)

            x1 = display_cnt[0][0] if (display_cnt[0][0] > display_cnt[1][0]) else display_cnt[1][0]
            y1 = display_cnt[0][1] if (display_cnt[0][1] > display_cnt[3][1]) else display_cnt[3][1]

            x2 = display_cnt[2][0] if (display_cnt[2][0] < display_cnt[3][0]) else display_cnt[3][0]
            y2 = display_cnt[1][1] if (display_cnt[1][1] < display_cnt[2][1]) else display_cnt[2][1]
            img = img[y1: y2, x1: x2]
            break

    return error, img



'''

All following functions are currently unused in this implementation.
They are only being kept here for reference and future experimentation.

'''

def page_border_alternative(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours and sort for largest contour
    cnts = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    status = -1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # Check if the contour starts on any border and is sufficiently large
        if x == 0 and y == 0 and w > self.img.shape[1]//2 and h > self.img.shape[0]//2:
            status = 1
            break

        # If the contour starts on any border and is sufficiently large
        if x+w == self.img.shape[1] and y+h == self.img.shape[0] and w > self.img.shape[1]//2 and h > self.img.shape[0]//2:
            status = 2
            break

        # If the area is essentially the whole page, it probably isn't the page
        if h*w > 0.97*self.img.shape[0]*self.img.shape[1]:
            status = 3
            break

        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check if a rectangle
        if len(approx) == 4:
            status = 0
            display_cnt = approx.reshape(4, 2)

            x1 = display_cnt[0][0] if (display_cnt[0][0] > display_cnt[1][0]) else display_cnt[1][0]
            y1 = display_cnt[0][1] if (display_cnt[0][1] > display_cnt[3][1]) else display_cnt[3][1]

            x2 = display_cnt[2][0] if (display_cnt[2][0] < display_cnt[3][0]) else display_cnt[3][0]
            y2 = display_cnt[1][1] if (display_cnt[1][1] < display_cnt[2][1]) else display_cnt[2][1]

            # Alternative way to border the page
            # This transforms the perspective, but doesn't border very well...
            from imutils.perspective import four_point_transform
            img = four_point_transform(img, display_cnt)
            break

    return status, img

# ==============================================================================
# The following functions are used to crop a border based on line intersections
# ==============================================================================


# Get the distance between two points
def point_distance(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

# Calculate the intersection of two lines
def line_intersection(line1, line2):
    if len(line1) == 4:
        line1 = [(line1[0], line1[1]), (line1[2], line1[3])]
        line2 = [(line2[0], line2[1]), (line2[2], line2[3])]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

# Calculate the distances between a point and the image corners
def corner_distances(shape, point):
    dist1 = self.point_distance((0, 0), point)
    dist2 = self.point_distance((shape[1] - 1, 0), point)
    dist3 = self.point_distance((0, shape[0] - 1), point)
    dist4 = self.point_distance((shape[1] - 1, shape[0] - 1), point)
    return dist1, dist2, dist3, dist4

# Crops the border additionally based on finding page edges
def crop_additional_border(img):

    # Create a mask of saturated pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > 20
    mask = mask.astype(np.uint8)*255

    # Perform canny and dilation
    c = (40, 50)
    edges = cv2.Canny(mask, c[0], c[1], apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(16, 16))
    edges = cv2.dilate(edges, kernel)

    # Find lines using HoughLinesP
    display_img = edges.copy()
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

    minLineLength = ((edges.shape[0]+edges.shape[1])/2)//5
    maxLineGap = ((edges.shape[0]+edges.shape[1])/2)//10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength, maxLineGap)

    # Find intersection points between the lines
    if lines is not None:
        lines = lines[:, 0, :]
        for x1, y1, x2, y2 in lines:
            cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.diagonal = self.point_distance(
            (0, 0), (edges.shape[1] - 1, edges.shape[0] - 1))
        dist = np.full((len(lines), len(lines), 4), self.diagonal*2)
        for e1, l in enumerate(lines):
            for e2, l in enumerate(lines):
                if e1 != e2:
                    inter = self.line_intersection(lines[e1], lines[e2])
                    # Checks to see if there is any intersection at all
                    if inter is not None:
                        # Intersection must occur within the image border
                        if inter[0] > 0 and inter[0] < edges.shape[1] and inter[1] > 0 and inter[1] < edges.shape[0]:
                            cv2.circle(display_img, inter,
                                       6, (0, 0, 255), -1)
                            dist[e1][e2][0], dist[e1][e2][1], dist[e1][e2][2], dist[e1][e2][3] = self.corner_distances(
                                edges.shape, inter)

        # Get the indices of the lines with the best intersection with minimum distance to each corner
        corner0_dist = dist[:, :, 0]
        corner1_dist = dist[:, :, 1]
        corner2_dist = dist[:, :, 2]
        corner3_dist = dist[:, :, 3]

        inter0_index = np.where(corner0_dist == corner0_dist.min())[0]
        inter1_index = np.where(corner1_dist == corner1_dist.min())[0]
        inter2_index = np.where(corner2_dist == corner2_dist.min())[0]
        inter3_index = np.where(corner3_dist == corner3_dist.min())[0]

        inter0_dist = dist[inter0_index[0], inter0_index[1], 0]
        inter1_dist = dist[inter1_index[0], inter1_index[1], 1]
        inter2_dist = dist[inter2_index[0], inter2_index[1], 2]
        inter3_dist = dist[inter3_index[0], inter3_index[1], 3]

        dist_thresh = (edges.shape[0]+edges.shape[1])//25

        if inter0_dist < dist_thresh:
            inter0 = self.line_intersection(
                lines[inter0_index[0]], lines[inter0_index[1]])
        else:
            inter0 = (0, 0)

        if inter1_dist < dist_thresh:
            inter1 = self.line_intersection(
                lines[inter1_index[0]], lines[inter1_index[1]])
        else:
            inter1 = (edges.shape[1], 0)

        if inter2_dist < dist_thresh:
            inter2 = self.line_intersection(
                lines[inter2_index[0]], lines[inter2_index[1]])
        else:
            inter2 = (0, edges.shape[0])

        if inter3_dist < dist_thresh:
            inter3 = self.line_intersection(
                lines[inter3_index[0]], lines[inter3_index[1]])
        else:
            inter3 = (edges.shape[1], edges.shape[0])

        left = max([inter0[0], inter2[0]])
        bottom = min([inter2[1], inter3[1]])
        right = min([inter1[0], inter3[0]])
        top = max([inter0[1], inter1[1]])
        return img[top:bottom, left:right]
    
    return img