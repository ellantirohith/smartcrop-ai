import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorExtractor:
    """
    Extracts dominant background colors from edge sampling points 
    using K-Means clustering.
    """
    def __init__(self, n_colors: int = 3):
        self.n_colors = n_colors

    def get_dominant_colors(self, image: np.ndarray) -> tuple:
        """
        Extracts dominant background color from image margins.
        Returns RGB color tuple.
        """
        h, w, _ = image.shape

        # Sample border pixels (outer 10%) to isolate background colors
        top = image[0:max(1, int(h*0.1)), :, :]
        bottom = image[min(h-1, int(h*0.9)):h, :, :]
        left = image[:, 0:max(1, int(w*0.1)), :]
        right = image[:, min(w-1, int(w*0.9)):w, :]

        border_pixels = np.vstack([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ])

        # Convert BGR to RGB for processing
        border_pixels_rgb = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

        kmeans = KMeans(n_clusters=self.n_colors, n_init=5, random_state=42)
        kmeans.fit(border_pixels_rgb)

        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        dominant_color = colors[np.argmax(counts)]

        return tuple(map(int, dominant_color))