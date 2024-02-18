import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def _plot_kmeans(cluster_states: list[np.ndarray],
                 cluster_centers: list[np.ndarray],
                 xlabel: str = "First Principal Component",
                 ylabel: str = "Second Principal Component",
                 zlabel: str = "Third Principal Component",
                 title: str = "",
                 plot_centers=True):
    """
    Function to plot clustered data

    :param cluster_states: an array with the cluster arrays
    :param cluster_centers: center coordinates of the clusters (-> plot_centers)
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param zlabel: z-axis label
    :param title: title of the plot
    :param plot_centers: whether to plot the centers as black dots or not
    """
    dim = cluster_states[0].shape[1]

    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
        for i, cluster in enumerate(cluster_states):
            ax.scatter(cluster[:, 0], cluster[:, 1])
            if plot_centers:
                center = cluster_centers[i]
                ax.scatter(center[0], center[1], c="black", marker=".", s=5)
                ax.annotate(str(i), (center[0], center[1]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif dim == 3:
        ax = fig.add_subplot(projection="3d")

        for i, cluster in enumerate(cluster_states):
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2])

            x, y, z = cluster[:, 0], cluster[:, 1], cluster[:, 2]

            mi = min([min(x), min(y), min(z)])
            ma = max([max(x), max(y), max(z)])
            ax.set_xlim([mi, ma])
            ax.set_ylim([mi, ma])
            ax.set_zlim([mi, ma])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

    plt.title(title)
    if dim == 2:
        plt.axis("square")
    plt.show()


def _plot_pca(data: np.ndarray,
              xlabel: str = "First Principal Component",
              ylabel: str = "Second Principal Component",
              zlabel: str = "Third Principal Component",
              title: str = "",
              square=True):
    """
    Function to plot PCA-reduced data set

    :param data: the data to plot (2 or 3-dimensional array)
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param zlabel: z-axis label
    :param title: title of the plot
    :param square: whether to scale axes equal or not
    """
    assert len(data.shape) == 2
    if data.shape[1] == 2:
        # Origin lines
        plt.axvline(0, color="black", alpha=0.3)
        plt.axhline(0, color="black", alpha=0.3)
        # Plot data
        plt.scatter(data[:, 0], data[:, 1])
        # Labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if square:
            plt.axis("square")
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot data
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
        # Labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    plt.title(title)
    plt.show()


def _concatenate_stories(words: list[list[str]], hidden_states: list[np.ndarray]) -> tuple[list[str], np.ndarray]:
    """
    Concatenate words and hidden states to one dataset

    :param words: list of word lists of the stories
    :param hidden_states: array containing hidden states of GPT2
    :return: concatenated word list with stacked hidden states;
    size: no. of all words, hidden_states: (#stories * #words, 13, 1, 768)
    """
    res_words = []
    for i in words:
        res_words += i
    res_hidden_states = np.concatenate(hidden_states, axis=0)

    return res_words, res_hidden_states


def _split_hidden_states(hidden_states: np.ndarray, stories: list[str]) -> list[np.ndarray]:
    """
    Split hidden states to arrays matching the stories => order of points hasn't changed

    :param hidden_states: states to split into story segments
    :param stories: the corresponding stories to split after
    """
    word_count = [len(s.split()) for s in stories]
    indices = [sum(word_count[0:i + 1]) for i in range(len(word_count))][:-1]
    return np.split(hidden_states, indices)


def convert_hidden_states_to_list(hidden_states: list[np.ndarray], layers=None, dims=None) -> \
        list[tuple[str, list[np.ndarray]]]:
    """
    Converts the plain hidden states to a feature list

    :param hidden_states: the hidden states as a list of numpy arrays
    :param layers: a list containing the layers to compute, defaults to [0, ..., 12]
    :param dims: a list containing the dimensions to compute, defaults to [0, ..., 767]
    :return: a list containing tuple objects (name + list of values)
    """
    # Put all tasks (name + values) in a list
    if layers is None:
        layers = [*range(13)]
    if dims is None:
        dims = [*range(768)]

    tasks = []

    # All features:
    for layer in layers:
        for dim in dims:
            feat_name = f"gpt2_L{layer}_d{dim}"
            feat_values = []
            # extract values from each story
            for story in hidden_states:
                feat_values.append(story[:, layer, dim])
            tasks.append((feat_name, feat_values))
    return tasks


def pca_hidden_states_to_list(hidden_states: list[np.ndarray], stories: list[str], pca_components: int, layers=None) \
        -> list[tuple[str, list[np.ndarray]]]:
    """
    Compute the Principal Component Analysis of the hidden states and converts them to features for the TRF

    :param hidden_states: states to transform
    :param stories: they are needed to split the states after the PCA in the corresponding story parts
    :param pca_components: number of desired principal components
    :param layers: a list of the layers to compute the PCA of (defaults to [0, ..., 12])
    :return: a list containing tuple objects (name + list of values)
    """
    # Put all tasks (name + values) in a list
    if layers is None:
        layers = [*range(13)]

    tasks = []

    words, concat_hidden_states = _concatenate_stories([i.split() for i in stories], hidden_states)

    for layer in layers:
        pca = PCA(n_components=pca_components)
        pca_hs = pca.fit_transform(concat_hidden_states[:, layer, :])

        print(f"Layer {layer}: {sum(pca.explained_variance_ratio_):.4}")

        # Optional plot of the PCA
        # _plot_pca(pca_hs, title=f"PCA with {pca_components} components: Layer {layer}")

        for dim in range(pca_components):
            feat_name = f"gpt2_pca{pca_components}dims_L{layer}_d{dim}"
            feat_values = _split_hidden_states(pca_hs[:, dim], stories)
            tasks.append((feat_name, feat_values))
    return tasks


def kmeans_pca_hidden_states_to_list(hidden_states: list[np.ndarray],
                                     stories: list[str],
                                     pca_components: int,
                                     n_cluster: int,
                                     layers=None) -> list[tuple[str, list[np.ndarray]]]:
    """
    Compute PCA and cluster the hidden states afterwards. Converts the result to a feature list for the TRF

    :param hidden_states: states to transform
    :param stories: they are needed to split the states after the PCA in the corresponding story parts
    :param pca_components: number of desired principal components
    :param n_cluster: number of desired clusters for the K-Means algorithm
    :param layers: a list of the layers to compute the PCA of (defaults to [0, ..., 12])
    :return: a list containing tuple objects (name + list of values)
    """
    # Put all tasks (name + values) in a list
    if layers is None:
        layers = [*range(13)]

    tasks = []

    words, concat_hidden_states = _concatenate_stories([i.split() for i in stories], hidden_states)
    for layer in layers:
        # PCA
        pca = PCA(n_components=pca_components)
        pca_hs = pca.fit_transform(concat_hidden_states[:, layer, :])

        kmeans = KMeans(n_cluster, n_init=10)
        mask = kmeans.fit_predict(pca_hs)
        # lists to keep results
        cluster_states = []
        cluster_indices = np.zeros(len(pca_hs))
        for i in range(n_cluster):
            indices = np.array(mask) == i
            cluster_indices[indices] = i
            cluster_states.append(pca_hs[indices])

        # Optional plot of the clusters
        # _plot_kmeans(cluster_states, kmeans.cluster_centers_,
        #             title=f"PCA with {pca_components} components: Layer {layer} ({n_cluster} clusters)")

        feat_name = f"gpt2_kmeans{n_cluster}clusters_pca{pca_components}dims_L{layer}_clusteridx"
        feat_values = _split_hidden_states(cluster_indices, stories)
        tasks.append((feat_name, feat_values))
    return tasks

# TODO: other preprocessing methods
