import numpy as np
import cv2
import math

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def process_word_image(cropped_org, i, j):
    ret, thresh = cv2.threshold(
        cropped_org, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, ksize=(2, 2))
    thresh = cv2.erode(thresh, kernel)

    ret = 1
    gray1 = thresh
    gray1[gray1 > ret] = 232
    gray1[gray1 < ret] = 155

    # We can do the whole 0.97 thing, but we should restrict it the a certain domain (like the connected components?) such
    # that we don't pick up anything from other lines (or lines themselves, hopefully)
    mean = cropped_org.mean()
    ret = mean * 0.95                    
    gray2 = thresh.copy()
    gray2[cropped_org > ret] = 232
    gray2[cropped_org < ret] = 155
    return gray1, gray2

def space_detection(thresh, i):
    col_sums = thresh.shape[0] - \
        thresh.sum(axis=0).astype(np.float32)/255

    # Get indices where there are no, or few, pixels
    data = np.argwhere(col_sums <= thresh.shape[0]/40)[:, 0]

    # Get consequtive runs of no black pixels
    consequtive = np.split(data, np.where(np.diff(data) != 1)[0]+1)

    # Characterize certain consequtive runs of white columns as spaces
    # Get the start and end of each gap, except for those on the edge of the image, then get the ranges/middle of these gaps
    consequtive_middle = []
    if len(consequtive[0]) > 0:

        consequtive = [(c[0], c[-1]) for c in consequtive if c[0]
                        != 0 and c[-1] != thresh.shape[1]-1]
        consequtive_middle = [(c[0] + c[1])//2 for c in consequtive]
        consequtive_range = np.array(
            [c[1] - c[0] for c in consequtive])
        
        max_gaps_indices = np.argsort(np.array(consequtive_range))
        sorted_range = consequtive_range[max_gaps_indices]
        range_diff = np.ediff1d(sorted_range)
        greater = np.argwhere(range_diff >= 8)[:, 0]

        if len(greater) == 0:
            top_k = len(sorted_range)
        else:
            top_k = greater[0] + 1

        gaps_accepted = max_gaps_indices[top_k:]
        consequtive_middle = np.array(consequtive_middle)[
                                      gaps_accepted]


    # Sometimes handwriting is slanted, so take the 'column sums' across a diagonal
    # Every x pixels, draw a white line across the thresh and see how many pixels it removes
    thresh_copy = thresh.copy()
    line_count = 100
    every_x = int(thresh.shape[1]/line_count)
    theta = 120 * 3.14 / 180.0
    length = thresh.shape[0]/math.sin(theta)

    changes = np.zeros(line_count)
    for i in range(line_count):
        x1 = i*every_x
        y1 = 0
        x2 =  int(x1 + length * math.cos(theta))
        y2 =  int(y1 + length * math.sin(theta))

        before = np.count_nonzero(thresh_copy)
        thresh_copy = cv2.line(thresh_copy, (x1, y1), (x2, y2), 255)
        changes[i] = np.count_nonzero(thresh_copy) - before

    # Observed: Must be at least 2 consequtive lines for there to be a space
    runs = zero_runs(changes)[1:]
    ranges = runs[:, 1] - runs[:, 0]
    runs_filtered = runs[np.argwhere(ranges >= 2)[:, 0]]
    start = runs_filtered[:, 0]
    end = runs_filtered[:, 1]
    middle_runs = start + (end - start)/2

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    for index in middle_runs:
        x1 = int(index*every_x)
        y1 = 0
        x2 =  int(x1 + length * math.cos(theta))
        y2 =  int(y1 + length * math.sin(theta))
        thresh = cv2.line(thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)

        x_mid =  int(x1 + length/2 * math.cos(theta))
        y_mid =  int(y1 + length/2 * math.sin(theta))
        cv2.circle(thresh, (x_mid, y_mid), 5, (0, 0, 255), -1)

    # Save line image for display
    for c in consequtive_middle:
        thresh = cv2.line(thresh, (c, 0), (c, thresh.shape[0]), (255, 0, 0), 2)

    return len(consequtive[0]) > 0, consequtive_middle, thresh
   
def get_words_in_line(img, dilated, components, line_components):
    words = []
    line_imgs = []
    # Process each line each line in line components
    for i, line in enumerate(line_components):
        words.append([])
        left_line = components.left[line]
        right_line = components.right[line]
        top_line = components.top[line]
        bottom_line = components.bottom[line]

        l = left_line.min()
        r = right_line.max()
        t = top_line.min()
        b = bottom_line.max()

        cropped_line = np.zeros(dilated.shape)

        for j in line:
            cropped_line[components.output == components.allowed[j] + 1] = 255
        cropped_line = cropped_line[t:b, l:r]
        thresh = cv2.bitwise_not(cropped_line.astype(np.uint8))


        # Find spaces to separate words
        accept, consequtive_middle, line_img = space_detection(thresh, i)

        if accept:
            line_imgs.append(line_img)
            
            # Determine which components are in each segment
            x_line = components.x[line]
            consequtive_middle = consequtive_middle + l
            consequtive_middle = np.append(
                consequtive_middle, dilated.shape[1])
            consequtive_middle.sort()

            segment = 0
            segments = [[] for i in range(len(consequtive_middle))]
            for x_ind in range(len(x_line)):
                segments[np.argmax(consequtive_middle > x_line[x_ind])].append(
                    line_components[i][x_ind])

            # Crop to the components in each word
            for j, s in enumerate(segments):
                left_seg = components.left[s]
                right_seg = components.right[s]
                top_seg = components.top[s]
                bottom_seg = components.bottom[s]
        
                if len(left_seg) > 0:
                    l = left_seg.min()
                    r = right_seg.max()
                    t = top_seg.min()
                    b = bottom_seg.max()

                    # Some horizontal line pieces can make it to this stage, so check for 'word'
                    # images which are very vertical and have small bounding area (e.g. the
                    # average component area - not to be confused with word area)
                    horizontal_line = (r - l)/(b - t) > 3 and (r - l)*(b - t) < components.bounding_area.mean()
                    if not horizontal_line:
                        cropped_org = cv2.cvtColor(
                            img[t:b, l:r], cv2.COLOR_RGB2GRAY)

                        word1, word2 = process_word_image(cropped_org, i, j)
                    
                        words[-1].append([word1, word2])
    return words, line_imgs
