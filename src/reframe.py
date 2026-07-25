import cv2
import numpy as np
from PIL import Image

class CanvasReframer:
    """
    Adjusts canvas aspect ratio and pads empty spaces with matching dominant background colors.
    """
    ASPECT_RATIOS = {
        "1:1 (Square / Post)": (1, 1),
        "16:9 (Landscape / Banner)": (16, 9),
        "9:16 (Portrait / Story)": (9, 16),
        "4:5 (Social Feed)": (4, 5)
    }

    def reframe(self, image: np.ndarray, subject_box: tuple, bg_color: tuple, target_ratio_key: str) -> Image.Image:
        """
        Pads original asset onto new aspect ratio canvas and overlays subject bounding box.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        target_w_ratio, target_h_ratio = self.ASPECT_RATIOS.get(target_ratio_key, (1, 1))

        target_aspect = target_w_ratio / target_h_ratio
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = w
            new_h = int(w / target_aspect)
        else:
            new_h = h
            new_w = int(h * target_aspect)

        # Build blank canvas initialized with dominant background color
        canvas = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)

        # Center original image on canvas
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        canvas[offset_y:offset_y + h, offset_x:offset_x + w] = img_rgb

        # Highlight detected subject area
        x1, y1, x2, y2 = subject_box
        cv2.rectangle(canvas, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), (0, 255, 0), 3)

        return Image.fromarray(canvas)