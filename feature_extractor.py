"""
Geometric Feature Extraction for Self-Learning Digit Recognition
Extracts 15 key geometric features that help identify digit patterns
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


class GeometricFeatureExtractor:
    """Extract geometric features from digit images for pattern recognition"""

    def __init__(self):
        self.feature_names = [
            'circle_completeness',     # Full circles (0, 8) vs partial (6, 9)
            'triangle_score',          # Incomplete triangles (7, 4)
            'vertical_lines',          # Vertical strokes (1, 4, 7)
            'horizontal_lines',        # Horizontal bars (7, 4, 5)
            'diagonal_strokes',        # Diagonal lines (7, 2)
            'hole_count',              # Enclosed regions: 8=2, 0/4/6/9=1
            'vertical_symmetry',       # Symmetry score (0, 8)
            'endpoint_count',          # Stroke endpoints (1, 7)
            'curvature_score',         # Smooth curves (2, 3, 5, 8)
            'top_density',             # Top-heavy (7, 9)
            'bottom_density',          # Bottom-heavy (2)
            'upper_loop',              # Loop in upper half (8, 9)
            'lower_loop',              # Loop in lower half (6, 8)
            'aspect_ratio',            # Height/width (1 is tall)
            'opening_direction',       # Left/right opening (5, 6, 9)
        ]

    def extract_all_features(self, image):
        """Extract all 15 geometric features from a 28x28 digit image"""
        # Ensure proper format
        if image.max() <= 1.0:
            binary = (image > 0.3).astype(np.uint8)
            normalized = image
        else:
            binary = (image > 76).astype(np.uint8)
            normalized = image / 255.0

        features = [
            self._circle_completeness(binary),
            self._triangle_score(binary),
            self._vertical_lines(binary),
            self._horizontal_lines(binary),
            self._diagonal_strokes(binary),
            self._hole_count(binary),
            self._vertical_symmetry(binary),
            self._endpoint_count(binary),
            self._curvature_score(binary),
            self._top_density(binary),
            self._bottom_density(binary),
            self._upper_loop(binary),
            self._lower_loop(binary),
            self._aspect_ratio(binary),
            self._opening_direction(binary),
        ]

        return np.array(features, dtype=np.float32)

    def _circle_completeness(self, binary):
        """Measure how complete circles are (0, 8 = high; 6, 9 = medium)"""
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if not regions:
            return 0.0

        max_circularity = 0.0
        for region in regions:
            if region.area > 10:
                perimeter = region.perimeter
                if perimeter > 0:
                    # Circularity: 1.0 = perfect circle
                    circularity = 4 * np.pi * region.area / (perimeter ** 2)
                    max_circularity = max(max_circularity, circularity)

        return max_circularity

    def _triangle_score(self, binary):
        """Detect triangle shapes (7 = high, 4 = medium)"""
        # Triangle: top-heavy with angular decrease
        h, w = binary.shape
        sections = 7
        densities = []

        for i in range(sections):
            start = i * h // sections
            end = (i + 1) * h // sections
            section_density = np.sum(binary[start:end, :]) / (w * (end - start))
            densities.append(section_density)

        if not densities or max(densities) == 0:
            return 0.0

        # Check if top-heavy and decreasing
        top_heavy = densities[0] > np.mean(densities)
        decreasing_trend = sum(densities[i] > densities[i+1] for i in range(len(densities)-1))

        score = 0.0
        if top_heavy:
            score += 0.5
        score += (decreasing_trend / (sections - 1)) * 0.5

        return score

    def _vertical_lines(self, binary):
        """Detect vertical strokes (1 = very high, 4, 7 = medium)"""
        # Vertical edge detection
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edges = ndimage.convolve(binary.astype(float), kernel)

        vertical_strength = np.sum(np.abs(edges)) / binary.size

        # Count continuous vertical columns
        vertical_cols = 0
        for col in range(binary.shape[1]):
            if np.sum(binary[:, col]) > 8:  # At least 8 pixels
                vertical_cols += 1

        return (vertical_strength + vertical_cols / binary.shape[1]) / 2

    def _horizontal_lines(self, binary):
        """Detect horizontal bars (7, 4, 5 have strong horizontal lines)"""
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        edges = ndimage.convolve(binary.astype(float), kernel)

        horizontal_strength = np.sum(np.abs(edges)) / binary.size

        # Count continuous horizontal rows
        horizontal_rows = 0
        for row in range(binary.shape[0]):
            if np.sum(binary[row, :]) > 8:
                horizontal_rows += 1

        return (horizontal_strength + horizontal_rows / binary.shape[0]) / 2

    def _diagonal_strokes(self, binary):
        """Detect diagonal lines (7, 2 have strong diagonals)"""
        diag1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        diag2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        conv1 = ndimage.convolve(binary.astype(float), diag1)
        conv2 = ndimage.convolve(binary.astype(float), diag2)

        return (np.sum(conv1) + np.sum(conv2)) / (2 * binary.size)

    def _hole_count(self, binary):
        """Count enclosed holes (8=2, 0/4/6/9=1, others=0)"""
        inverted = 1 - binary
        labeled = measure.label(inverted, connectivity=1)
        regions = measure.regionprops(labeled)

        holes = 0
        h, w = binary.shape
        for region in regions:
            bbox = region.bbox
            touches_border = (bbox[0] == 0 or bbox[1] == 0 or
                            bbox[2] == h or bbox[3] == w)

            if not touches_border and region.area > 5:
                holes += 1

        return float(min(holes, 3))  # Cap at 3

    def _vertical_symmetry(self, binary):
        """Measure left-right symmetry (0, 8 = high)"""
        flipped = np.fliplr(binary)
        difference = np.sum(np.abs(binary - flipped))
        max_difference = binary.size

        symmetry = 1.0 - (difference / max_difference)
        return symmetry

    def _endpoint_count(self, binary):
        """Count stroke endpoints (1 has 2, 7 has 1-2)"""
        if np.sum(binary) < 10:
            return 0.0

        skeleton = morphology.skeletonize(binary)

        # Count neighbors for each skeleton pixel
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0

        neighbor_count = ndimage.convolve(skeleton.astype(float), kernel, mode='constant')
        endpoints = np.sum((skeleton > 0) & (neighbor_count == 1))

        return float(endpoints) / 5.0  # Normalize

    def _curvature_score(self, binary):
        """Measure smoothness of curves (2, 3, 5, 8 = high)"""
        laplacian = ndimage.laplace(binary.astype(float))
        curvature = np.sum(np.abs(laplacian)) / binary.size
        return min(curvature, 1.0)

    def _top_density(self, binary):
        """Pixel density in top third (7, 9 = high)"""
        top = binary[:9, :]
        return np.sum(top) / top.size

    def _bottom_density(self, binary):
        """Pixel density in bottom third (2 = high)"""
        bottom = binary[19:, :]
        return np.sum(bottom) / bottom.size

    def _upper_loop(self, binary):
        """Detect loop in upper half (8, 9 = high)"""
        upper = binary[:14, :]
        inverted = 1 - upper
        labeled = measure.label(inverted, connectivity=1)
        regions = measure.regionprops(labeled)

        for region in regions:
            bbox = region.bbox
            touches_border = (bbox[0] == 0 or bbox[1] == 0 or
                            bbox[3] == upper.shape[1])

            if not touches_border and region.area > 5:
                return 1.0

        return 0.0

    def _lower_loop(self, binary):
        """Detect loop in lower half (6, 8 = high)"""
        lower = binary[14:, :]
        inverted = 1 - lower
        labeled = measure.label(inverted, connectivity=1)
        regions = measure.regionprops(labeled)

        for region in regions:
            bbox = region.bbox
            touches_border = (bbox[1] == 0 or bbox[2] == lower.shape[0] or
                            bbox[3] == lower.shape[1])

            if not touches_border and region.area > 5:
                return 1.0

        return 0.0

    def _aspect_ratio(self, binary):
        """Height to width ratio (1 = tall, 0 = wide)"""
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if not regions:
            return 1.0

        largest = max(regions, key=lambda r: r.area)
        bbox = largest.bbox

        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        if width == 0:
            return 2.0

        return min(height / width, 3.0)  # Cap at 3

    def _opening_direction(self, binary):
        """Detect C-shape opening direction (0=left, 0.5=both, 1=right)"""
        h, w = binary.shape
        left_col = binary[:, :w//3]
        right_col = binary[:, 2*w//3:]
        middle = binary[:, w//3:2*w//3]

        left_density = np.sum(left_col) / left_col.size
        right_density = np.sum(right_col) / right_col.size
        middle_density = np.sum(middle) / middle.size

        if left_density < 0.1 and right_density < 0.1:
            return 0.5  # No opening

        # If left is empty but right is full = opens left (like 6)
        # If right is empty but left is full = opens right (like 9)
        if middle_density > 0.2:
            if left_density > right_density * 1.5:
                return 0.0  # Opens left
            elif right_density > left_density * 1.5:
                return 1.0  # Opens right

        return 0.5  # Balanced or closed

    def get_feature_count(self):
        """Return number of features"""
        return len(self.feature_names)


def extract_features_batch(images):
    """
    Extract features from multiple images

    Args:
        images: array of shape (n, 28, 28)

    Returns:
        features: array of shape (n, 15)
    """
    extractor = GeometricFeatureExtractor()
    features = []

    for img in images:
        feat = extractor.extract_all_features(img)
        features.append(feat)

    return np.array(features)