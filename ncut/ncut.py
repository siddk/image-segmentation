"""
ncut.py

Core class and runner for Normalized Cuts Image Segmentation. Constructs a Region Adjacency Graph
(RAG) over an image, and then recursively performs Normalized Cuts on it to segment the image.

Inspired by skimage implementation of Normalized Cuts.
"""
from skimage import segmentation, color
from skimage.future import graph
import cv2
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
        :return: Cluster labels
        """
        return segmentation.slic(self.image, n_segments=num_clusters, compactness=compactness)

    def construct_rag(self, labels):
        """
        Construct the region adjacency graph using the Super-Pixel (labels) as graph nodes.

        :param labels: Super-Pixel Labels.
        :return: Region-Adjacency-Graph (networkx graph)
        """
        return graph.rag_mean_color(self.image, labels, mode='similarity')

if __name__ == "__main__":
    # Build the segmenter
    args = sys.argv
    segmenter = NCutSegment(args[1])

    # Get the super-pixeled image
    cluster_labels = segmenter.cluster(segmenter.image.shape[0], 30)
    super_pixels = color.label2rgb(cluster_labels, segmenter.image, kind='avg')
    border_image = segmentation.mark_boundaries(super_pixels, cluster_labels, (0, 0, 0))

    # Construct the Region Adjacency Graph over the Super-Pixeled Image
    g = segmenter.construct_rag(super_pixels)

    # Do the Normalized Cuts
    n_cuts = graph.ncut(cluster_labels, g)
    segmented_image = color.label2rgb(n_cuts, segmenter.image, kind='avg')

    cv2.imshow("Super Pixels", border_image)
    cv2.imshow("N-Cuts", segmented_image)
    cv2.waitKey(0)