"""
kmeans.py

Core class and runner for K-Means image segmentation code. Given an image and a number of clusters,
run the K-Means clustering algorithm to partition the image by color, and return the respective
clusters.
"""
import cv2
import numpy as np
import sys


class KMeansSegment():
    def __init__(self, image, num_segments):
        """
        Instantiate a K-Means Image Segmenter with the given image and the number of clusters to
        partition.

        :param image: Path to the given image.
        :param num_segments: Number of clusters to partition.
        """
        self.image = cv2.imread(image)
        self.num_segments = num_segments

    def segment(self):
        """
        Run the K-Means algorithm to segment the image.

        :return: Tuple of label matrix, result matrix
        """
        # Blur the image (for better results)
        image = cv2.GaussianBlur(self.image, (7, 7), 0)

        # Reshape the image into color (RGB) vectors
        vectors = np.float32(image.reshape(-1, 3))

        # Setup K-Means run criteria and get K-Means results
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(vectors, self.num_segments, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)

        # Build the segmented image by swapping the labels with cluster centers (re-colorize image)
        segmented_image = centers[labels.flatten()].reshape(image.shape)
        return labels.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)


if __name__ == "__main__":
    # Build the segmenter
    args = sys.argv
    segmenter = KMeansSegment(args[1], int(args[2]))

    # Show the original image, and the segmented image
    label, segmented = segmenter.segment()
    cv2.imshow("Original", segmenter.image)
    cv2.imshow("Segmented", segmented)
    cv2.waitKey(0)

