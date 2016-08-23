"""
ncut.py

Core class and runner for Normalized Cuts Image Segmentation. Constructs a Region Adjacency Graph
(RAG) over an image, and then recursively performs Normalized Cuts on it to segment the image.

Inspired by skimage implementation of Normalized Cuts.
"""
from scipy import sparse
from scipy.sparse import linalg
from skimage import segmentation, color
from skimage.future import graph
import cv2
import networkx as nx
import numpy as np
import sys


class NCutSegment():
    def __init__(self, image):
        """
        Instantiate the NCut Segmenter with the path to the given image.

        :param image: Path to the image to segment.
        """
        self.image = cv2.imread(image)

    def cluster(self, num_clusters, compactness):
        """
        Run loose K-Means clustering over the image, to construct super-pixels for use in creating
        the Region-Adjacency-Graph (RAG).

        :param num_clusters: Number of super-pixels to partition.
        :param compactness: How much to weight space/color proximity (higher --> Space proximity)
        :return: Numpy Matrix same shape as the original image, with cluster labels.
        """
        return segmentation.slic(self.image, compactness=compactness, n_segments=num_clusters)

    def construct_rag(self, labels):
        """
        Construct the region adjacency graph using the Super-Pixel (labels) as graph nodes.

        :param labels: Super-Pixel Labels.
        :return: NetworkX Graph representing the Region-Adjacency-Graph.
        """
        return graph.rag_mean_color(self.image, labels, mode='similarity')

    def ncut(self, labels, rag, thresh=0.001, num_cuts=10, max_edge=1.0):
        """
        Perform the N-Cut Algorithm given the super-pixels, and the RAG (Graph).

        :param labels: Labels for each super-pixel (RAG Node)
        :param rag: Actual NX Graph, with Nodes and Weighted Edges
        :param thresh: Graph won't be further divided if the value of the n-cut exceeds this.
        :param num_cuts: Number of N-Cuts to perform before determining the optimal cut.
        :param max_edge: Maximum possible edge value (i.e. for a self-edge)
        :return: Numpy Matrix same shape as the original image, with segment labels.
        """
        for node in rag.nodes_iter():
            rag.add_edge(node, node, weight=max_edge)

        self.ncut_recurse(rag, thresh, num_cuts)

        map_array = np.zeros(labels.max() + 1, dtype=labels.dtype)

        # Mapping from old labels to new
        for n, d in rag.nodes_iter(data=True):
            map_array[d['labels']] = d['ncut label']

        return map_array[labels]

    def ncut_recurse(self, rag, thresh, num_cuts):
        """
        Perform N-Cut on Graph, by recursively partitioning the graph into two until the value of
        the Cut exceeds thresh, or no more cuts can be completed.

        Credit: skimage.future.graph._ncut

        :param rag:
        :param thresh:
        :param num_cuts:
        :return:
        """
        d, w = self.dw_matrices(rag)
        m = w.shape[0]

        if m > 2:
            d2 = d.copy()
            # Since d is diagonal, we can directly operate on its data the inverse of square root
            d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

            # Refer Shi & Malik 2001, Equation 7, Page 891
            vals, vectors = linalg.eigsh(d2 * (d - w) * d2, which='SM', k=min(100, m - 2))

            # Pick second smallest eigenvector. Refer Shi & Malik 2001, Section 3.2.3, Page 893
            vals, vectors = np.real(vals), np.real(vectors)
            index2 = self.argmin2(vals)
            ev = vectors[:, index2]

            cut_mask, mcut = self.get_min_ncut(ev, d, w, num_cuts)
            if mcut < thresh:
                # Sub divide and perform N-cut again Refer Shi & Malik 2001, Section 3.2.5, Page 893
                sub1, sub2 = self.partition_by_cut(cut_mask, rag)

                self.ncut_recurse(sub1, thresh, num_cuts)
                self.ncut_recurse(sub2, thresh, num_cuts)
                return

        # The N-cut wasn't small enough, or could not be computed. The remaining graph is a region.
        # Assign `ncut label` by picking any label from the existing nodes, since `labels` are
        # unique, `new_label` is also unique.
        node = rag.nodes()[0]
        new_label = rag.node[node]['labels'][0]
        for n, d in rag.nodes_iter(data=True):
            d['ncut label'] = new_label

    def argmin2(self, arr):
        """
        Given an array, return the index of the second smallest element.

        :param arr: Array of elements
        :return: Index of second smallest element.
        """
        min1, min2 = np.inf, np.inf
        min_idx1, min_idx2 = 0, 0
        for i in range(len(arr)):
            x = arr[i]
            if x < min1:
                min2, min_idx2 = min1, min_idx1
                min1, min_idx1 = x, i
            elif min1 < x < min2:
                min2, min_idx2 = x, i
        return min_idx2

    def get_min_ncut(self, ev, d, w, num_cuts):
        """
        Threshold an eigenvector evenly, to determine minimum ncut.

        :param ev: The eigenvector to threshold
        :param d: The RAG's diagonal matrix
        :param w: The RAG's edge weight matrix
        :param num_cuts: The number of evenly spaced thresholds to check for
        :return Tuple of mask, mcut, where
            mask: The array of booleans which denotes the bi-partition.
            mcut: The value of the minimum ncut
        """
        mcut, mn, mx = np.inf, ev.min(), ev.max()

        # If all values in `ev` are equal, it implies that the graph can't be further sub-divided.
        # In this case the bi-partition is the the graph itself and an empty set.
        min_mask = np.zeros_like(ev, dtype=np.bool)
        if np.allclose(mn, mx):
            return min_mask, mcut

        # Refer Shi & Malik 2001, Section 3.1.3, Page 892 Perform evenly spaced n-cuts and determine
        # the optimal one.
        for t in np.linspace(mn, mx, num_cuts, endpoint=False):
            mask = ev > t
            cost = self.ncut_cost(mask, d, w)
            if cost < mcut:
                min_mask = mask
                mcut = cost

        return min_mask, mcut

    def partition_by_cut(self, cut, rag):
        """
        Compute resulting subgraphs from given bi-partition.

        :param cut: Array of booleans, 'True' elements belong to one set.
        :param rag: The Region Adjacency Graph.
        :return sub1, sub2, the subgraphs from the partition.
        """
        nodes1 = [n for i, n in enumerate(rag.nodes()) if cut[i]]
        nodes2 = [n for i, n in enumerate(rag.nodes()) if not cut[i]]
        return rag.subgraph(nodes1), rag.subgraph(nodes2)

    def ncut_cost(self, cut, d, w):
        """
        Returns the N-cut cost of a bi-partition of a graph.

        :param cut: The mask for the nodes in the graph. 'True' nodes are in one set.
        :param d: The diagonal matrix of the graph.
        :param w: The weight matrix of the graph.
        :return The cost of performing the N-cut.
        """
        cut = np.array(cut)
        cut_cost = self.cut_cost(cut, w)

        # D has elements only along the diagonal, one per node, so we can directly
        # index the data attribute with cut.
        assoc_a = d.data[cut].sum()
        assoc_b = d.data[~cut].sum()

        return (cut_cost / assoc_a) + (cut_cost / assoc_b)

    def cut_cost(self, cut, w):
        """
        Return the total weight of crossing edges in a bi-partition.

        :param cut: Array of booleans, with 'True' elements belonging to one set.
        :param w: The weight matrix of the graph.
        :return Returns the total weight of crossing edges.
        """
        num_rows, num_cols = w.shape[0], w.shape[1]
        indices, indptr = w.indices, w.indptr
        cost = 0.0

        for col in range(num_cols):
            for row_index in range(indptr[col], indptr[col + 1]):
                row = indices[row_index]
                if cut[row] != cut[col]:
                    cost += w.data[row_index]

        return cost * 0.5

    def dw_matrices(self, g):
        """
        Return the diagonal and weight matrices of a graph.

        :param graph: A Region Adjacency Graph (NetworkX)
        :return Tuple (D, W) where:
            D: The Diagonal Matrix of the Graph (D[i, i] is the sum of weights of all
               edges incident on node 'i')
            W: The Weight Matrix of the Graph (W[i, j] is the weight of the edge joining 'i' to 'j')
        """
        w = nx.to_scipy_sparse_matrix(g, format='csc')
        d = sparse.dia_matrix((w.sum(axis=0), 0), shape=w.shape).tocsc()
        return d, w

if __name__ == "__main__":
    # Build the segmenter
    args = sys.argv
    segmenter = NCutSegment(args[1])

    # Get the super-pixeled image
    cluster_labels = segmenter.cluster(segmenter.image.shape[0], 30)
    super_pixels = color.label2rgb(cluster_labels, segmenter.image, kind='avg')
    border_image = segmentation.mark_boundaries(super_pixels, cluster_labels, (0, 0, 0))

    # Construct the Region Adjacency Graph over the Super-Pixeled Image
    g = segmenter.construct_rag(cluster_labels)

    # Do the Normalized Cuts
    n_cuts = segmenter.ncut(cluster_labels, g)
    segmented_image = color.label2rgb(n_cuts, segmenter.image, kind='avg')

    cv2.imshow("Original", segmenter.image)
    cv2.imshow("Super Pixels", border_image)
    cv2.imshow("N-Cuts", segmented_image)
    cv2.waitKey(0)