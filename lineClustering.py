import numpy as np
import cv2

# Arrange data into groups where successive elements differ by no more than maxgap (Assumes data is sorted)
def clustering(data, maxgap):
    groups = [[data[0]]]
    indices = [[0]]
    for i, x in enumerate(data[1:]):
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
            indices[-1].append(i+1)
        else:
            groups.append([x])
            indices.append([i+1])

    return groups, indices

# Clusters objects into horizontal lines
def line_clustering(components):#y, cX, cY, c_area, boundingRect, allowed_area, output, display_borders, display_img):

    # Get top k areas
    top_k_vals = min([10, len(components)])
    max_areas = (-np.sort(-components.area))[:top_k_vals].mean()

    # Filter contours with 'small enough' area
    c_indices = np.argwhere(components.area >= max_areas/10)
    c_area = components.area[c_indices].reshape((c_indices.shape[0]))
    cX = components.x[c_indices].reshape((c_indices.shape[0]))
    cY = components.y[c_indices].reshape((c_indices.shape[0]))
    boundingRect = components.bounding_rect[c_indices].reshape((c_indices.shape[0], 4))

    sorted_indices = cY.argsort(axis=0)
    c_area = c_area[sorted_indices]
    cX = cX[sorted_indices]
    cY = cY[sorted_indices]
    boundingRect = boundingRect[sorted_indices]
    mean_height = boundingRect[:, 3].mean()

    # Cluster the bounding boxes together
    clusters, cluster_indices = list(clustering(list(cY), mean_height/1.5))
    cluster_means = [int(sum(c)/len(c)) for c in clusters]
    
    # Now that we've found the cluster x values, assign components to each cluster based on x
    diffs = [cluster_means[i+1] - cluster_means[i]
        for i in range(len(cluster_means)-1)]
    avg_diff = sum(diffs)/len(diffs)

    component_clusters = np.zeros(len(components))
    component_clusters_min_dist = np.zeros(len(components))

    for i in range(len(components)):
        clusters_diff = np.array([abs(components.y[i] - c) for c in cluster_means])
        component_clusters[i] = np.argmin(clusters_diff)
        component_clusters_min_dist[i] = clusters_diff[int(component_clusters[i])]

    # Create list of components belonging to each line
    line_components = []
    for c in range(len(clusters)):
        keep_components = np.zeros((components.output.shape))
        line_components.append([])
        for i in range(len(components)):
            if component_clusters[i] == c and component_clusters_min_dist[i] < avg_diff/2:
                keep_components[components.output == components.allowed[i] + 1] = 255
                line_components[-1].append(i)

    # Filter out lines with very little area
    lines = [i for i, l in enumerate(line_components) if
                       components.area[np.array(l)].sum() >= components.min_area*5]
    line_components = [l for i, l in enumerate(line_components) if i in lines]
    clusters = [c for i, c in enumerate(clusters) if i in lines]
    cluster_means = [m for i, m in enumerate(cluster_means) if i in lines]

    # Create display image
    keep_components = np.zeros((components.output.shape))
    for c in range(len(clusters)):
        for i in range(len(components)):
            if component_clusters[i] == c and component_clusters_min_dist[i] < avg_diff/2:
                keep_components[components.output == components.allowed[i] + 1] = 255

    for i, cc in enumerate(cluster_means):
        cv2.line(keep_components, (0, cluster_means[i]), (
            keep_components.shape[1], cluster_means[i]), 255, 3)

    return line_components, keep_components
