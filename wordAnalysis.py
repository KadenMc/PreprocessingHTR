import numpy as np
import cv2

def crop_to_black(img):
    """Crops a 2D image, where each pixel is 0 or 255, to the black pixels."""
    black = np.argwhere(img == 0)
    t = black[:, 0].min()
    b = black[:, 0].max()
    l = black[:, 1].min()
    r = black[:, 1].max()
    return img[t:b, l:r]


def threshold_otsu(img, thresh_multiplier=None, color1=255, color2=0):
    """Threshold a greyscale image using otsu thresholding. Uses some erosion."""
    ret, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, ksize=(2, 2))
    thresh = cv2.erode(thresh, kernel)
    thresh[thresh >= 1] = color1
    thresh[thresh < 1] = color2
    return thresh


def threshold_mean(img, thresh_multiplier=0.95, color1=255, color2=0):
    """Threshold a greyscale image using mean thresholding."""
    mean = img.mean()
    ret = mean * thresh_multiplier
    img[img > ret] = color1
    img[img < ret] = color2
    return img


def threshold_multiple(img, thresh_func, count, thresh_multiplier=0.95, color1=255, color2=0):
    """
    Splits an image into multiple sections, thresholding these sections, and re-combines them
    into a final thresholded image. This is more robust to changes in light across the image.
    """
    w = int(np.round(img.shape[1]/(count + 1)))
    for i in range(count + 1):
        img[:, i*w:(i + 1)*w] = thresh_func(img[:, i*w:(i + 1)*w],
                                                   thresh_multiplier=thresh_multiplier,
                                                   color1=color1, color2=color2)
    return img


def threshold_multiple_line(img, thresh_func, page_width, thresh_multiplier=0.95, color1=255, color2=0):
    """
    Utilizes threshold_multiple, first determining an appropriate image split count
    for that line size. The wider the line, the larger the count.
    """
    count = int(np.round((img.shape[1]/page_width)*15))
    return threshold_multiple(img, thresh_func, count, thresh_multiplier=thresh_multiplier, color1=color1, color2=color2)


def remove_vertical_components(components, ind):
    """Removes vertically skinny components, which are often unwanted lines/artifacts in the image."""
    w = components.right[ind] - components.left[ind]
    h = components.bottom[ind] - components.top[ind]

    # Return components with an acceptable h/w ratio
    return ind[np.argwhere(h/w < 7)[:, 0]]


def clean_line_thresh(img, consider_location=True, area_multiplier=1):
    """Cleans a thresholded image using morphing and connected components."""

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove smallest components with connected component analysis
    import connectedComponentsProcessing as ccp

    # Create components
    components = ccp.Components(255-img)

    # Intuition for consider_location: Small components near an y mean weighted by area are more
    # likely to be disconnected letter segments, small whereas components near the botton/top of
    # the image are much more likely to be noise!
    if consider_location:

        # Take a weighted mean of the y value, where the component area is the weight
        y_weighted_mean = np.average(components.y, weights=components.area)

        # Get each component's y distance from the weighted mean
        dist = np.abs(components.y - y_weighted_mean)

        # Squash this into a proportion
        dist = dist/max([y_weighted_mean, img.shape[0] - y_weighted_mean])
        
        min_area = ccp.mean_top_k(components.area, k=15)/8
        allowed = np.argwhere(((1 - dist)**2)*components.bounding_area > min_area * area_multiplier)[:, 0]
    else:

        min_area = ccp.mean_top_k(components.area, k=15)/3
        allowed = np.argwhere(components.bounding_area > min_area * area_multiplier)[:, 0]

    img = np.zeros((img.shape))
    for i in range(len(allowed)):
        img[components.output == allowed[i] + 1] = 255
    img = 255 - img
    
    return img


def roll_zero_pad(a, shift, axis=None):
    """Shifts an array left if shift < 0 and right if shift > 0, padding the new elements with 0."""
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def get_gap_from_sums(sums, thresh):
    """Returns interval gaps in the row or column sums of an image. Ignores 'edge' gaps.

    Parameters:
    sums (np.ndarray): An array (dtype == np.bool) representing whether there is any pixels in a column.
    """
    data = np.argwhere(sums != 0)[:, 0]
    consequtive = np.split(data, np.where(np.diff(data) != 1)[0] + 1)

    if len(consequtive[0]) == 0:
        return []

    # Characterize consequtive runs of white columns as gaps
    # Get the start and end of each gap, but ignore any gaps on the edge of an image
    return [[c[0], c[-1]] for c in consequtive if c[0] != 0 and c[-1] != thresh.shape[1]-1]


def get_sums(img):
    """Determine whether there are any black pixels in each column of an image.

    Parameters:
    img (np.ndarray): An 2D image where each pixel has value 0 or 255.
    """
    return np.invert(((255-img).sum(axis=0)).astype(np.bool))


def get_gaps(img, degree_slant=30):
    """Gets gaps in an image, a slanted verion of the image, and the intersection of those gaps."""
    height = img.shape[0]
    width = img.shape[1]

    # Get gaps in image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sums = get_sums(img.astype(np.uint8))
    gaps = get_gap_from_sums(sums, img)

    #cv2.imshow("img_line", np.tile(sums.astype(np.uint8)*255, (100, 1)))

    # Get gaps in slanted image - image is slanted theta degrees
    # This is very necessary for people who write with a forward slant
    theta = np.radians(degree_slant)
    rolled = img.copy()
    for i in range(height):
        hyp = int(np.round(i*np.tan(theta)))
        rolled[i, :] = roll_zero_pad(rolled[i, :], hyp)

    '''cv2.imshow('rolled', rolled)'''

    sums_slanted = get_sums(rolled)

    #cv2.imshow("sums_slanted", np.tile(sums_slanted.astype(np.uint8)*255, (100, 1)))

    # Shift slanted image - Slanted 45 degree line '/'. Calculate the subtraction to
    # bring the gaps back in line with those original image
    subtract = int((height * np.cos((np.pi/2) - theta))//2)
    sums_slanted = roll_zero_pad(sums_slanted, -subtract)
    gaps_slanted = get_gap_from_sums(sums_slanted, rolled)

    #cv2.imshow("sums_slanted-subtracted", np.tile(sums_slanted.astype(np.uint8)*255, (100, 1)))

    # Get intersection of gaps in the image and the slanted image
    sums_both = (np.logical_and(sums_slanted.astype(np.bool), sums.astype(np.bool)))
    gaps_both = get_gap_from_sums(sums_both, img)

    #cv2.imshow("both_line", np.tile(sums_both.astype(np.uint8)*255, (100, 1)))

    return gaps, gaps_slanted, gaps_both


# Use gaps in lines to determine a suitable minimum gap between two words
def get_min_gap(lines, page_width):
    """
    Determines a minimum gap which exists between words. Under the right circumstances, gaps larger
    than this should be considered spaces.
    """
    widths = np.array([lines[i].right - lines[i].left for i in range(len(lines))])
    gaps_all = [lines[i].gaps for i in range(len(lines))]
    gaps_slanted_all = [lines[i].gaps_slanted for i in range(len(lines))]
    gaps_both_all = [lines[i].gaps_both for i in range(len(lines))]

    # Get line width proportion to page width and add 16%, which is about the max size of a border
    # This gives us how much of the page this line takes up
    line_width_proportions = widths/page_width + 0.16

    # Multiplying these by the average words per line (10), gives us an expected word count
    #expected_words = line_width_proportions*10
    # Now, we'll adjust the expected words based on the text and space size

    # Generally, there is at most 11 words per line, so if the line proportion is 1, we'll take the
    # top 10 spaces for a full line
    min_gap = 0
    count = 0
    for g, gaps in enumerate(gaps_all):
        if len(gaps) != 0:
            k = int(np.ceil(line_width_proportions[g] * 10) - 1)
            gaps = np.array(gaps)
            ranges = gaps[:, 1] - gaps[:, 0]
            ranges.sort()

            # Don't count lines with less than 3 expected words, since they may have just one word,
            # this would mess min_gap to count them!
            if k > 3:
                min_gap += ranges[-k:].mean()
                count += 1

    # No lines w/ words/gaps detected... assume all lines have one word, so use a massive min_gap
    if count == 0:
        return page_width

    return int(np.round(min_gap/count))


def find_nearest_value_index(array, value):
    """Finds the nearest value to those in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_middle_delete_unmatched(middles1, middles2, min_gap):
    """Takes two arrays of space dividers and removes dividers from middles1 which aren't in middles2."""
    if len(middles1) == 0 or len(middles2) == 0:
        return []

    middles1 = np.array(middles1)
    middles2 = np.array(middles2)
    nearest = np.array([find_nearest_value_index(middles2, m) for m in middles1])
    diff = np.abs(middles2[nearest] - middles1)
    middles_final = middles1[np.argwhere(diff < min_gap)[:, 0]]
    return middles_final


def filter_middles(gaps, min_gap):
    """Filters gaps smaller than some minimum gap threshold."""
    middles = [(g[0] + g[1])//2 for g in gaps]
    ranges = [g[1] - g[0] for g in gaps]
    return [m for i, m in enumerate(middles) if ranges[i] > min_gap]


def get_middle(img, gaps, gaps_slanted, gaps_both, min_gap):
    """Calculates reasonable space dividers in a line (middles) provided its gaps."""
    # Get middles
    middles = filter_middles(gaps, min_gap)
    middles_slanted = filter_middles(gaps_slanted, min_gap*1.13)
    middles_both = filter_middles(gaps_both, min_gap*0.88)

    # Draw middles
    display_img = img.copy().astype(np.float32)
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

    for m in middles:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (255, 0, 0), 2)

    for m in middles_slanted:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (0, 255, 0), 2)

    for m in middles_both:
        display_img = cv2.line(display_img, (m, 0), (m, display_img.shape[0]), (0, 0, 255), 2)

    # Merge the multiple analyses into one
    middles.extend(middles_slanted)
    middles.extend(middles_both)
    middles_merged = np.array(middles)
    middles_merged.sort()

    if len(middles_merged) == 0:
        return [], img, img

    merge_sum = middles_merged[0]
    merge_count = 1
    middles_final = []
    for i in range(1, len(middles_merged)):
        if middles_merged[i] - middles_merged[i - 1] < min_gap:
            merge_sum += middles_merged[i]
            merge_count += 1
        else:
            middles_final.append(int(np.round(merge_sum/merge_count)))
            merge_sum = middles_merged[i]
            merge_count = 1
    
    middles_final.append(int(np.round(merge_sum/merge_count)))  

    for c in middles_final:
        display_img = cv2.line(display_img, (c-4, 0), (c-4, display_img.shape[0]), 0, 2)
        display_img = cv2.line(display_img, (c+4, 0), (c+4, display_img.shape[0]), 0, 2)

    return middles_final, display_img, img


class Word():
    """
    Holds information about a word in the text.

    Attributes
    ----------
    left : int
        The left border of the word in the image.
    right : int
        The right border of the word in the image.
    top : int
        The top border of the word in the image.
    bottom : int
        The bottom border of the word in the image.
    words : list[np.ndarray]
        A list of images of the word.
    """

    def __init__(self, images, left, right, top, bottom):
        self.images = images
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

class Line():
    """
    Holds information about (and performs operations on) a line of text.

    Attributes
    ----------
    left : int
        The left border of the line in the image.
    right : int
        The right border of the line in the image.
    top : int
        The top border of the line in the image.
    bottom : int
        The bottom border of the line in the image.
    words : list[Word]
        A list of Word objects containing information about each word.

    Methods
    -------
    get_middles():
        Get space dividers (middles) for the line.
    
    crop_words():
        Segments a line image into word images.
    """

    def __init__(self, components, line, img, config):
        self.components = components

        # Remove artifact components which are easier to detect in the context of a line
        self.line = remove_vertical_components(components, line)

        if len(self.components.left[self.line]) == 0:
            self.valid = False
            return

        self.valid = True
        
        # Extract line bounding box information
        self.left = self.components.left[self.line].min()
        self.right = self.components.right[self.line].max()
        self.top = self.components.top[self.line].min()
        self.bottom = self.components.bottom[self.line].max()

        # Create connected components image
        self.comp_img = np.zeros(img.shape)
        for j in self.line:
            self.comp_img[self.components.output == self.components.allowed[j] + 1] = 255
        self.comp_img = cv2.bitwise_not(self.comp_img[self.top:self.bottom, self.left:self.right].astype(np.uint8))

        config['save_inter_func'](config, self.comp_img, "line_comp")

        # Create a (multiple) thresholded image
        self.thresh_img = cv2.cvtColor(img[self.top:self.bottom, self.left:self.right], cv2.COLOR_BGR2GRAY)
        self.thresh_img = threshold_multiple_line(self.thresh_img, threshold_mean, img.shape[1])
        self.thresh_img = clean_line_thresh(self.thresh_img)

        config['save_inter_func'](config, self.thresh_img, "line_thresh")

        # Calculate gaps for connected components and thresholded images
        self.gaps, self.gaps_slanted, self.gaps_both = get_gaps(self.comp_img)
        self.gaps_thresh, self.gaps_slanted_thresh, self.gaps_both_thresh = get_gaps(self.thresh_img)


    def get_middles(self, min_gap, config):
        '''Get space dividers (middles) for the line.'''
        # Get middles for components
        self.middles, display_img, thresh = get_middle(self.comp_img, self.gaps, self.gaps_slanted,
                                            self.gaps_both, min_gap)

        for m in self.middles:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        config['save_inter_func'](config, display_img, "line_separated1")

        # Get middles for thresholded
        self.middles_thresh, display_img, thresh = get_middle(self.thresh_img, self.gaps_thresh,
                                                          self.gaps_slanted_thresh, self.gaps_both_thresh,
                                                          min_gap)

        for m in self.middles_thresh:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        config['save_inter_func'](config, display_img, "line_separated2")

        # If the thresholded image doesn't have a line where the components image does, it should be removed
        # This is because the thresholded image has less missing text, so middles could created via missing text
        middles_final = get_middle_delete_unmatched(self.middles, self.middles_thresh, min_gap)

        display_img = thresh.copy()
        for m in middles_final:
            display_img = cv2.line(display_img, (m, 0), \
                                   (m, display_img.shape[0]), (0, 255, 0), 3)
        config['save_inter_func'](config, display_img, "line_separated_final")

        return middles_final, thresh


    def crop_words(self, img, min_gap, config):
        '''Segments a line image into word images.'''
        self.middles_final, thresh = self.get_middles(min_gap, config)
        self.words = []

        # If there is no gaps, create just one word
        if len(self.middles_final) == 0:
            segments = [self.line]

        # Otherwise, separate the components into lines
        else:
        
            # Determine which components are in each segment
            x_line = self.components.x[self.line] - self.left
            self.middles_final = np.append(self.middles_final, img.shape[1])
            self.middles_final.sort()

            segments = [[] for i in range(len(self.middles_final))]
            for x_ind in range(len(x_line)):
                segments[np.argmax(self.middles_final > \
                                   x_line[x_ind])].append(self.line[x_ind])

        # Crop to the components in the line
        for j, s in enumerate(segments):
            left_seg = self.components.left[s]
            right_seg = self.components.right[s]
            top_seg = self.components.top[s]
            bottom_seg = self.components.bottom[s]

            if len(left_seg) > 0:
                l = left_seg.min()
                r = right_seg.max()
                t = top_seg.min()
                b = bottom_seg.max()

                # Create word images
                word1 = thresh[t - self.top:b - self.top, \
                               l - self.left:r - self.left]
                word1 = crop_to_black(word1).astype(np.uint8)

                word2 = cv2.cvtColor(img[t:b, l:r], cv2.COLOR_RGB2GRAY)
                word2 = threshold_multiple(word2, threshold_otsu, 4)
                word2 = clean_line_thresh(word2, consider_location=True, \
                                          area_multiplier=2).astype(np.uint8)

                # Recolor word images for SimpleHTR
                word1[word1 == 0] = 155
                word1[word1 == 255] = 232

                word2[word2 == 0] = 155
                word2[word2 == 255] = 232
                
                self.words.append(Word([word1, word2], l, r, t, b))


def get_words_in_line(img, components, line_components, config):
    '''Segments a line image into word images.'''
    # Process each line
    lines = []
    for i, line in enumerate(line_components):
        line_obj = Line(components, line, img, config)
        if line_obj.valid:
            lines.append(line_obj)

    # Use gaps in lines to determine a suitable minimum gap between two words
    min_gap = get_min_gap(lines, img.shape[1])

    # Crop lines into words
    words = []
    for i in range(len(lines)):
        lines[i].crop_words(img, min_gap, config)

    return lines