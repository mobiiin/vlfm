from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert


class ObjectDetections:
    """
    Provides a consistent format for object detections generated by both object
    detection and grounding models.
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        phrases: List[str],
        image_source: Optional[np.ndarray] = None,
        visualize: bool = False,
        fmt: str = "cxcywh",
    ):
        self.image_source = image_source
        self.boxes = box_convert(boxes=boxes, in_fmt=fmt, out_fmt="xyxy")
        self.logits = logits
        self.phrases = phrases
        if visualize:
            self.annotated_frame = annotate(
                image_source=image_source,
                boxes=self.boxes,
                logits=logits,
                phrases=phrases,
            )
        else:
            self.annotated_frame = None

    def filter_by_conf(self, conf_thresh: float):
        """Filters detections by confidence threshold in-place.

        Args:
            conf_thresh (float): Confidence threshold to filter detections.
        """

        keep: torch.Tensor[bool] = torch.ge(self.logits, conf_thresh)  # >=

        self.boxes = self.boxes[keep]
        self.logits = self.logits[keep]
        self.phrases = [p for i, p in enumerate(self.phrases) if keep[i]]

        if self.annotated_frame is not None:
            # Re-visualize with filtered detections
            self.annotated_frame = annotate(
                image_source=self.image_source,
                boxes=self.boxes,
                logits=self.logits,
                phrases=self.phrases,
            )

    def to_json(self) -> dict:
        """
        Converts the object detections to a JSON serializable format.

        Returns:
            dict: A dictionary containing the object detections.
        """
        return {
            "boxes": self.boxes.tolist(),
            "logits": self.logits.tolist(),
            "phrases": self.phrases,
        }

    @classmethod
    def from_json(
        cls,
        json_dict: dict,
        image_source: Optional[np.ndarray] = None,
        visualize: bool = False,
    ):
        """
        Converts the object detections from a JSON serializable format.

        Args:
            json_dict (dict): A dictionary containing the object detections.
            image_source (Optional[np.ndarray], optional): Optionally provide the
                original image source. Defaults to None.
            visualize (bool, optional): A flag indicating whether to visualize the
                output data. Defaults to False.
        """
        return cls(
            image_source=image_source,
            boxes=torch.tensor(json_dict["boxes"]),
            logits=torch.tensor(json_dict["logits"]),
            phrases=json_dict["phrases"],
            visualize=visualize,
            fmt="xyxy",
        )


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
) -> np.ndarray:
    """
    Annotates an image with bounding boxes, class names, and scores.

    Args:
        image_source (np.ndarray): Input image in numpy array format.
        boxes (torch.Tensor): A tensor of shape (N, 4) containing the bounding boxes
            for each object in the image. The bounding boxes should be in the format
            (x1, y1, x2, y2).
        logits (torch.Tensor): A tensor of shape (N, C) containing the confidence
            scores for each object in the image.
        phrases (List[str]): A list of strings containing the class names for each
            object in the image.

    Returns:
        np.ndarray: The original image with the bounding boxes, class names, and
            scores labeled on it.
    """
    img = image_source.copy()

    # Draw bounding boxes, class names, and scores on image
    for box, prob, phrase in zip(boxes, logits, phrases):
        # Convert tensor to numpy array
        box = box.numpy()
        prob = prob.numpy()

        # If the box appears to be in normalized coordinates, de-normalize using the
        # image dimensions
        if box.max() <= 1:
            box = box * np.array(
                [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            )
            box = box.astype(int)

        # Draw bounding box
        point1 = (int(box[0]), int(box[1]))
        point2 = (int(box[2]), int(box[3]))
        img = draw_bounding_box(
            image=img,
            point1=point1,
            point2=point2,
            class_name=phrase,
            score=prob.max(),
        )

    return img


def draw_bounding_box(
    image: np.ndarray,
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    class_name: str,
    score: float,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draws a bounding box on an image and labels it with a class name and score.

    Args:
        image (np.ndarray): Input image in RGB numpy array format.
        point1 (Tuple[int, int]): The coordinates for the top left point of the bounding
            box.
        point2 (Tuple[int, int]): The coordinates for the bottom right point of the
            bounding box.
        class_name (str): A string representing the class name of the predicted object
            within the bounding box.
        score (float): A confidence score of the predicted object within the bounding
            box. This should be in the range of 0.0 to 1.0.
        color (Optional[Tuple[int, int, int]]): A tuple containing RGB values (in the
            range of 0-255) of the bounding box color. If None, the color will be
            randomly chosen.

    Returns:
        np.ndarray: The original image with the bounding box and corresponding class
            name and score labeled on it.
    """
    # Create a copy of the input image to draw on
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    if color is None:
        # Randomly choose a color from the rainbow colormap (so boxes aren't black)
        single_pixel = np.array([[np.random.randint(0, 256)]], dtype=np.uint8)
        single_pixel = cv2.applyColorMap(single_pixel, cv2.COLORMAP_RAINBOW)

        # reshape to a single dimensional array
        color = single_pixel.reshape(3)
    else:
        # Convert RGB to BGR
        color = color[::-1]
    color = [int(c) for c in color]

    # Draw bounding box on image
    box_thickness = 2
    cv2.rectangle(img, point1, point2, color, thickness=box_thickness)

    # Draw class name and score on image
    text_label = f"{class_name}: {int(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(text_label, font, font_scale, font_thickness)
    text_x = point1[0]
    text_y = point2[1] + text_size[1]
    cv2.rectangle(
        img,
        (text_x, text_y - 2 * text_size[1]),
        (text_x + text_size[0], text_y - text_size[1]),
        color,
        -1,
    )
    cv2.putText(
        img,
        text_label,
        (text_x, text_y - text_size[1] - box_thickness),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
