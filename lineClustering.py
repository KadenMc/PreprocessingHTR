import numpy as np
import cv2
import matplotlib.pyplot as plt


def piecewise_linear(x, x0, y0, k1, k2):
    """Define a piecewise, lienar function with two line segments."""
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


def determine_components_n(Ms, lower_bounds, all_means, config):
    """Determine the optimal number of GMM components based on loss."""

    """
    Explanation:
    
    Lower bounds looks somewhat like a piecewise function with 2 lines
    The changepoint from one line to another tends to be the correct
    number of GMM components!

    This makes sense since any additional components would help the
    error (lower bound) a lot less, leading to a much less steep line

    Then, the goal is to find this changepoint, which we can do by
    fitting a piecewise, 2-line function with scipy.optimize.curve_fit
    For whatever reason, method='trf' works best!
    """

    x = np.array([float(i) for i in range(len(lower_bounds))])
    y = np.array(lower_bounds)

    from scipy import optimize
    p, e = optimize.curve_fit(piecewise_linear, x, y, method='trf')

    if config['inter']:
        plt.xlabel('components')
        plt.ylabel('lower_bounds')
        plt.plot(x, y, 'o')
        x = np.linspace(x.min(), x.max(), 1000)
        plt.plot(x, piecewise_linear(x, *p))
        config['save_inter_func'](config, None, "gmm_components", plot=True)

    # p[0] is the changepoint parameter
    return int(np.round(p[0])) + 1
    

def gmm_clustering(cY, components, config):
    """Uses GMM models to cluster text lines based on their y values."""
    from sklearn.mixture import GaussianMixture
    Ms = list(range(1, 30))

    lower_bounds = []
    all_means = []
    for m in Ms:
        gmm = GaussianMixture(n_components=m, random_state=0).fit(np.expand_dims(cY, 1))
        lower_bounds.append(gmm.lower_bound_)        
        means = gmm.means_.squeeze()

        # Sort if multiple means, or turn into an array is just one
        try:
            means.sort()
        except:
            means = np.array([means])

        all_means.append(means)


    # Different methods for selecting the number of components
    n = determine_components_n(Ms, lower_bounds, all_means, config)

    # Perform analysis with determined number of components n
    gmm = GaussianMixture(n_components=n, random_state=0).fit(np.expand_dims(cY, 1))
    cluster_means = (gmm.means_.squeeze()).astype(np.int32)
    cluster_means.sort()
    return cluster_means


def line_clustering(components, config):
    """Clusters components into horizontal lines."""

    # Organize and sort component data by y values
    c_area = components.area
    cX = components.x
    cY = components.y
    boundingRect = components.bounding_rect
    
    sorted_indices = cY.argsort(axis=0)
    c_area = c_area[sorted_indices]
    cX = cX[sorted_indices]
    cY = cY[sorted_indices]
    boundingRect = boundingRect[sorted_indices]
    mean_height = boundingRect[:, 3].mean()

    # Perform GMM analysis to determine lines based on y values
    cluster_means = gmm_clustering(cY, components, config)
    
    # Now that we've found the cluster y values, assign components to each cluster based on y
    component_clusters = np.zeros(len(components))
    component_clusters_min_dist = np.zeros(len(components))

    cluster_i = 0
    line_components = [[]]
    component_clusters = []
    for i in range(len(cY)):
        if cluster_i < len(cluster_means) - 1:
            if abs(cY[i] - cluster_means[cluster_i]) > abs(cluster_means[cluster_i + 1] - cY[i]):
                cluster_i += 1
                line_components.append([])
        
        line_components[-1].append(i)
        component_clusters.append(cluster_i)
    
    component_clusters = np.array(component_clusters)

    # Convert the 'sorted y' indices back to the original component indices
    for i, l in enumerate(line_components):
        sorter = np.argsort(sorted_indices)
        line_components[i] = sorter[np.searchsorted(sorted_indices,
                                                    np.array(l), sorter=sorter)]

    # Filter out lines with very little area
    '''lines = [i for i, l in enumerate(line_components) if
                       components.area[l].sum() >= components.min_area*2]
    line_components = [l for i, l in enumerate(line_components) if i in lines]
    cluster_means = [m for i, m in enumerate(cluster_means) if i in lines]'''

    # Create display image
    keep_components = np.zeros((components.output.shape))
    for c in range(len(cluster_means)):
        for i in range(len(components)):
            if component_clusters[i] == c:
                keep_components[components.output == components.allowed[i] + 1] = 255

    for i, cc in enumerate(cluster_means):
        cv2.line(keep_components, (0, cluster_means[i]), (
            keep_components.shape[1], cluster_means[i]), 255, 3)

    config['save_inter_func'](config, keep_components, "lines")

    return line_components