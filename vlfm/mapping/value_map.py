# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_rotation_matrix
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

DEBUG = False
SAVE_VISUALIZATIONS = False
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"
RECORDING_DIR = "value_map_recordings"
JSON_PATH = osp.join(RECORDING_DIR, "data.json")
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")


class ValueMap(BaseMap):
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35 # Lower _decision_threshold to allow more updates to the value map.
    _map: np.ndarray

    def __init__(
        self,
        value_channels: int,
        size: int = 1000,
        use_max_confidence: bool = False,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        """
        Args:
            value_channels: The number of channels in the value map.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
        """
        if PLAYING:
            size = 2000
        super().__init__(size)
        self._value_map = np.zeros((size, size, value_channels), np.float32)
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map
        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        if RECORDING:
            if osp.isdir(RECORDING_DIR):
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # Dump all args to a file
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # Create a blank .json file inside for now
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)

    def _create_circular_mask(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # ignore
        """
        Create a circular mask centered at the robot's position, divided into 4 directional masks
        (forward, backward, left, right) based on the FOV. Each mask will be smaller than a quarter circle.

        Args:
            depth: The depth image to use for determining the visible portion of the FOV.
            tf_camera_to_episodic: The transformation matrix from the episodic frame to the camera frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in RADIANS.

        Returns:
            Four boolean masks for the forward, backward, left, and right directions.
        """
        # Get the robot's position in the map
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # Calculate the radius of the circle based on max_depth
        radius = int(max_depth * self.pixels_per_meter)  # Radius in pixels

        # Create a grid of coordinates relative to the robot's position
        y, x = np.ogrid[-px:self.size - px, -py:self.size - py]
        distance_sq = x**2 + y**2  # Squared distance from the center

        # Create a circular mask based on the radius
        circular_mask = distance_sq <= radius**2

        # Calculate the angle for each pixel relative to the robot's orientation
        angles = np.arctan2(y, x)  # Angles in radians, ranging from -pi to pi

        # Extract the robot's yaw (orientation) from the transformation matrix
        yaw = extract_yaw(tf_camera_to_episodic)

        # Adjust angles to be relative to the robot's orientation
        angles = angles - yaw

        # Normalize angles to the range [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        # Define the angular ranges for each mask based on the FOV
        half_fov = fov / 2  # Half of the FOV
        forward_angle_range = (-half_fov, half_fov)  # Forward mask
        # Backward mask (centered at π radians)
        # Ensure the range is continuous by splitting it into two parts if it crosses the -π/π boundary
        backward_angle_range_1 = (np.pi - half_fov, np.pi)
        backward_angle_range_2 = (-np.pi, -np.pi + half_fov)
        left_angle_range = (np.pi / 2 - half_fov, np.pi / 2 + half_fov)  # Left mask
        right_angle_range = (-np.pi / 2 - half_fov, -np.pi / 2 + half_fov)  # Right mask

        # Create masks for each direction based on the angular ranges
        forward_mask = np.logical_and(circular_mask, np.logical_and(angles >= forward_angle_range[0], angles <= forward_angle_range[1]))
        # Backward mask is split into two parts to handle angle wrapping
        backward_mask = np.logical_and(circular_mask, np.logical_or(
            np.logical_and(angles >= backward_angle_range_1[0], angles <= backward_angle_range_1[1]),
            np.logical_and(angles >= backward_angle_range_2[0], angles <= backward_angle_range_2[1])
        ))
        left_mask = np.logical_and(circular_mask, np.logical_and(angles >= left_angle_range[0], angles <= left_angle_range[1]))
        right_mask = np.logical_and(circular_mask, np.logical_and(angles >= right_angle_range[0], angles <= right_angle_range[1]))

        return forward_mask, backward_mask, left_mask, right_mask

    def update_map(
        self,
        values: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """Updates the value map with the given depth image, pose, and value to use.

        Args:
            values: The value to use for updating the map. This is now a numpy array of
                action scores for ["Go forward", "Go backward", "Turn right", "Turn left"].
            depth: The depth image to use for updating the map; expected to be already
                normalized to the range [0, 1].
            tf_camera_to_episodic: The transformation matrix from the episodic frame to
                the camera frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in RADIANS.
        """
        assert (
            len(values) == 4
        ), f"Incorrect number of values given ({len(values)}). Expected 4 (one for each action)."

        # Create a circular mask centered at the robot's position
        # forward_mask, backward_mask, left_mask, right_mask = self._create_circular_mask(
        # depth, tf_camera_to_episodic, min_depth, max_depth, fov
        # )

        # # Combine the masks into a dictionary for easier handling
        # masks = {
        #     "forward": forward_mask,
        #     "backward": backward_mask,
        #     "left": left_mask,
        #     "right": right_mask,
        # }

        # If an obstacle map is provided, mask out non-navigable and unexplored areas
        # if self._obstacle_map is not None:
        #     # Get the navigable area and explored area from the obstacle map
        #     # navigable_area = self._obstacle_map._navigable_map
        #     explored_area = self._obstacle_map.explored_area

        #     # Mask out non-navigable areas from the new map and masks
        #     for key in masks:
        #         masks[key][explored_area == 0] = 0
        #     # Mask out unexplored areas from the new map, confidence map, and value map
        #     self._map[explored_area == 0] = 0
        #     self._value_map[explored_area == 0] = 0  # Reset values in unexplored areas

        # for i, (_, mask) in enumerate(masks.items()):
        #     self._value_map[mask, 0] = values[i]  # "Go forward"
        
        curr_map, masks = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)
        # self.plot_and_save_localized_data(curr_map)
        # self.plot_and_save_masks(masks)

        # Fuse the new data with the existing data, taking obstacles into account
        self._fuse_new_data(values, masks, curr_map)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "values": values.tolist(),
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)


    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.
            radius (float): The radius in meters to use for selecting the best waypoint.
            reduce_fn (Callable, optional): The function to use for reducing the values
                within the given radius. Defaults to np.max.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple of the sorted waypoints and
                their corresponding values.
        """
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        values = [get_value(point) for point in waypoints]

        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(values)

        # Use np.argsort to get the indices of the sorted values
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """Return an image representation of the map"""
        # Must negate the y values to get the correct orientation
        reduced_map = reduce_fn(self._value_map).copy()
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        map_img = np.flipud(reduced_map)
        # Make all 0s in the value map equal to the max value, so they don't throw off
        # the color mapping (will revert later)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # Revert all values that were originally zero to white
        map_img[zero_mask] = (255, 255, 255)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

            if markers is not None:
                for pos, marker_kwargs in markers:
                    map_img = self._traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        cv2.imwrite("mapimage.png", map_img)
        return map_img


    def _process_local_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
        Returns:
            A mask of the visible portion of the FOV.
        """
        # Squeeze out the channel dimension if depth is a 3D array
        if len(depth.shape) == 3:
            depth = depth.squeeze(2)

        # Squash depth image into one row with the max depth value for each column
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row
        y = depth_row * np.tan(angles)

        # Get the dictionary of masks (forward, back, left, right)
        confidence_masks, masks_dict = self._get_confidence_mask(fov, max_depth)

        # Initialize the final visible mask
        visible_mask = np.zeros_like(list(confidence_masks.values())[0], dtype=np.uint8)

        # Process each mask in the dictionary
        for mask_name, cone_mask in confidence_masks.items():
            # Convert the x, y coordinates to pixel coordinates
            x_pixels = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
            y_pixels = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

            # Create a contour from the x, y coordinates, with the top left and right
            # corners of the image as the first two points
            last_row = cone_mask.shape[0] - 1
            last_col = cone_mask.shape[1] - 1
            start = np.array([[0, last_col]])
            end = np.array([[last_row, last_col]])
            contour = np.concatenate((start, np.stack((y_pixels, x_pixels), axis=1), end), axis=0)

            # Draw the contour onto the cone mask, in filled-in black
            mask = cv2.drawContours(cone_mask.copy(), [contour], -1, 0, -1)  # type: ignore

            # Combine the processed mask into the final visible_mask
            visible_mask = np.maximum(visible_mask, mask)

            if DEBUG:
                vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
                for point in contour:
                    vis[point[1], point[0]] = (0, 255, 0)
                if SAVE_VISUALIZATIONS:
                    # Create visualizations directory if it doesn't exist
                    if not os.path.exists("visualizations"):
                        os.makedirs("visualizations")
                    # Expand the depth_row back into a full image
                    depth_row_full = np.repeat(depth_row.reshape(1, -1), depth.shape[0], axis=0)
                    # Stack the depth images with the visible mask
                    depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    depth_row_full = cv2.cvtColor((depth_row_full * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    vis = np.flipud(vis)
                    new_width = int(vis.shape[1] * (depth_rgb.shape[0] / vis.shape[0]))
                    vis_resized = cv2.resize(vis, (new_width, depth_rgb.shape[0]))
                    vis = np.hstack((depth_rgb, depth_row_full, vis_resized))
                    time_id = int(time.time() * 1000)
                    cv2.imwrite(f"visualizations/{time_id}_{mask_name}.png", vis)
                else:
                    cv2.imshow(f"obstacle mask ({mask_name})", vis)
                    cv2.waitKey(0)
        # print(f"visible_mask {visible_mask.shape}")
        return visible_mask, masks_dict


    def _localize_new_data(
            self,
            depth: np.ndarray,
            tf_camera_to_episodic: np.ndarray,
            min_depth: float,
            max_depth: float,
            fov: float,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Updates the map with new data and returns the updated map along with rotated masks.

        Args:
            depth: The depth image to use for updating the map.
            tf_camera_to_episodic: The transformation matrix from the camera frame to the episodic frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in radians.

        Returns:
            A tuple containing:
            - The updated map (curr_map) as a numpy array.
            - A dictionary of rotated masks (masks_dict) with keys 'forward', 'left', 'right', 'backward'.
        """
        # Get new portion of the map and the masks dictionary
        curr_data, masks_dict = self._process_local_data(depth, fov, min_depth, max_depth)
        
        # Rotate this new data and masks to match the camera's orientation
        yaw = extract_yaw(tf_camera_to_episodic)
        if PLAYING:
            if yaw > 0:
                yaw = 0
            else:
                yaw = np.deg2rad(30)

        # Rotate the current data
        curr_data = rotate_image(curr_data, -yaw)

        # Rotate each mask in the masks_dict
        rotated_masks_dict = {}
        for mask_name, mask in masks_dict.items():
            rotated_masks_dict[mask_name] = rotate_image(mask, -yaw)
            
        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # Overlay the new data onto the map
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        for mask_name, rotated_mask in rotated_masks_dict.items():
            # Overlay the rotated mask onto the map at the specified position (px, py)
            rotated_masks_dict[mask_name] = place_img_in_img(np.zeros_like(self._map), rotated_mask, px, py).astype(bool)
        
        return curr_map, rotated_masks_dict

    # def _localize_new_data(
    #     self,
    #     depth: np.ndarray,
    #     tf_camera_to_episodic: np.ndarray,
    #     min_depth: float,
    #     max_depth: float,
    #     fov: float,
    #     ) -> np.ndarray:
    #     # Get new portion of the map
    #     curr_data = self._process_local_data(depth, fov, min_depth, max_depth)

    #     # Rotate this new data to match the camera's orientation
    #     yaw = extract_yaw(tf_camera_to_episodic)
    #     if PLAYING:
    #         if yaw > 0:
    #             yaw = 0
    #         else:
    #             yaw = np.deg2rad(30)
    #     curr_data = rotate_image(curr_data, -yaw)

    #     # Determine where this mask should be overlaid
    #     cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

    #     # Convert to pixel units
    #     px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
    #     py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

    #     # Overlay the new data onto the map
    #     curr_map = np.zeros_like(self._map)
    #     curr_map = place_img_in_img(curr_map, curr_data, px, py)

    #     return curr_map

    def plot_and_save_masks(self, masks: dict) -> None:
        """
        Plot and save 4 masks from a dictionary as a single image.

        Args:
            masks (dict): A dictionary of 4 masks with keys ["forward", "backward", "left", "right"].
                        Each mask is a 2D numpy array of shape (height, width).
            output_filename (str): The name of the output image file. Default is "combined_masks.png".
        """
        # Ensure there are exactly 4 masks
        assert len(masks.values()) == 4, "The dictionary must contain exactly 4 masks."

        # Extract masks from the dictionary
        mask_forward = masks["forward"]
        mask_backward = masks["backward"]
        mask_left = masks["left"]
        mask_right = masks["right"]

        # Ensure all masks have the same shape
        assert mask_forward.shape == mask_backward.shape == mask_left.shape == mask_right.shape, "All masks must have the same shape."

        # Scale masks to the range [0, 255] and convert to uint8
        mask_forward = (mask_forward * 255).astype(np.uint8)
        mask_backward = (mask_backward * 255).astype(np.uint8)
        mask_left = (mask_left * 255).astype(np.uint8)
        mask_right = (mask_right * 255).astype(np.uint8)

        # Overlay masks using logical OR (addition can also be used if you want to blend them)
        combined_image = cv2.bitwise_or(mask_forward, mask_backward)
        combined_image = cv2.bitwise_or(combined_image, mask_left)
        combined_image = cv2.bitwise_or(combined_image, mask_right)

        # Save the combined image
        cv2.imwrite("combined_masks.png", combined_image)

    def plot_and_save_localized_data(self, localized_data: np.ndarray, output_filename: str = "zzzzzzzzz.png") -> None:
        """
        Plots the output of the _localize_new_data function and saves it to an image.

        Args:
            localized_data: The output from the _localize_new_data function.
            output_filename: The name of the output image file. Defaults to "zzzzzzzzz.png".
        """
        # Normalize the data to the range [0, 255] for visualization
        localized_data_normalized = (localized_data * 255).astype(np.uint8)
        
        # Convert the grayscale image to a 3-channel RGB image for better visualization
        localized_data_rgb = cv2.cvtColor(localized_data_normalized, cv2.COLOR_GRAY2RGB)
        
        # Save the image to the specified filename
        cv2.imwrite(output_filename, localized_data_rgb)
        
            
    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        size = int(max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(  # type: ignore
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(fov) / 3 + 90,  # start_angle
            np.rad2deg(fov) / 3 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask


    def _get_confidence_mask(self, fov: float, max_depth: float):
        """Generate a FOV cone with central values weighted more heavily and return a dictionary of masks (forward, back, left, right)."""
        # if (fov, max_depth) in self._confidence_masks:
        #     return self._confidence_masks[(fov, max_depth)].copy()

        # Generate the forward mask
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
        forward_mask = adjusted_mask * cone_mask

        # Generate the back mask by rotating the forward mask 180 degrees
        back_mask = np.rot90(forward_mask, 2)

        # Generate the left mask by rotating the forward mask 90 degrees counterclockwise
        left_mask = np.rot90(forward_mask, 1)

        # Generate the right mask by rotating the forward mask 90 degrees clockwise
        right_mask = np.rot90(forward_mask, 3)

        confidence_masks = {
            "forward": forward_mask,
            "backward": back_mask,
            "left": left_mask,
            "right": right_mask,
        }

        # Threshold the masks to convert them to binary (0 or 1)
        forward_mask = (forward_mask > 0).astype(np.uint8)
        back_mask = (back_mask > 0).astype(np.uint8)
        left_mask = (left_mask > 0).astype(np.uint8)
        right_mask = (right_mask > 0).astype(np.uint8)

        # Store the masks in a dictionary
        masks_dict = {
            "forward": forward_mask,
            "backward": back_mask,
            "left": left_mask,
            "right": right_mask,
        }

        # # Cache the masks for future use
        # self._confidence_masks[(fov, max_depth)] = confidence_masks.copy()

        return confidence_masks, masks_dict

    def _fuse_new_data(
        self,
        values: np.ndarray,
        masks: Dict[str, np.ndarray],
        new_map: np.ndarray,
    ) -> None:
        """Fuse the new data with the existing value and confidence maps.

        Args:
            values: The values attributed to the new portion of the map.
            masks: A dictionary of boolean masks for ["forward", "backward", "left", "right"].
            new_map: The new map data to fuse. Confidences are between 0 and 1, with 1 being the most confident.
        """
        assert len(values) == 4, f"Incorrect number of values given ({len(values)}). Expected 4."

        # If an obstacle map is provided, mask out non-navigable and unexplored areas
        if self._obstacle_map is not None:
            explored_area = self._obstacle_map.explored_area
            for key in masks:
                masks[key][explored_area == 0] = 0
            new_map[explored_area == 0] = 0
            self._map[explored_area == 0] = 0
            self._value_map[explored_area == 0] = 0  # Reset values in unexplored areas

        if self._fusion_type == "replace":
            # Replace existing values with new values where new_map is greater
            for i, (key, mask) in enumerate(masks.items()):
                self._value_map[mask] = values[i]
            self._map[new_map > self._map] = new_map[new_map > self._map]

        elif self._fusion_type == "equal_weighting":
            # Use equal weighting to blend existing and new values
            for i, (key, mask) in enumerate(masks.items()):
                existing_values = self._value_map[mask]
                new_values = np.full_like(existing_values, values[i])
                blended_values = (existing_values + new_values) / 2.0
                self._value_map[mask] = blended_values
            self._map = (self._map + new_map) / 2.0

        else:
            # Default fusion: weighted average based on confidence
            sum_confidence = self._map + new_map
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weight_existing = self._map / sum_confidence
                weight_new = new_map / sum_confidence

            # Update confidence map
            self._map = self._map * weight_existing + new_map * weight_new

            weight_existing = np.repeat(np.expand_dims(weight_existing, axis=2), self._value_channels, axis=2)
            weight_new = np.repeat(np.expand_dims(weight_new, axis=2), self._value_channels, axis=2)
            # print(f"weight_existing shape {weight_existing.shape}")
            # print(f"weight_new shape  {weight_new.shape}")
            # print(f"_masksssss  {masks}")
            # print(f"self._value_map  {self._value_map}")

            for i, (key, mask) in enumerate(masks.items()):
                # print(f"_mask  {mask}")
                existing_values = self._value_map[mask]
                # value_map_image = (existing_values * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                # cv2.imwrite("existing_values.png", value_map_image)
                # print(f"_existing_values  {existing_values}")
                new_values = np.full_like(existing_values, values[i])
                self._value_map[mask] = (weight_existing[mask] * existing_values + weight_new[mask] * new_values)
                
            
            self._value_map = np.nan_to_num(self._value_map)
            self._map = np.nan_to_num(self._map)
            # print(f"self._map shape  {self._map.shape}")
            # print(f"self._value_map shape  {self._value_map.shape}")
            

        # value_map_image = (self._value_map * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        # cv2.imwrite("value_map.png", value_map_image)
        # _map_image = (self._map * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        # cv2.imwrite("_map.png", _map_image)

        
def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def replay_from_dir() -> None:
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    v = ValueMap(**kwargs)

    sorted_keys = sorted(list(data.keys()))

    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        values = np.array(data[img_path]["values"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map(
            values,
            depth,
            tf_camera_to_episodic,
            float(data[img_path]["min_depth"]),
            float(data[img_path]["max_depth"]),
            float(data[img_path]["fov"]),
        )

        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    if PLAYING:
        replay_from_dir()
        quit()

    v = ValueMap(value_channels=1)
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._process_local_data(
        depth=depth,
        fov=np.deg2rad(40),
        min_depth=0.5,
        max_depth=5.0,
    )
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)

    num_points = 20

    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map(
            np.array([1]),
            depth,
            tf,
            min_depth=0.5,
            max_depth=5.0,
            fov=np.deg2rad(79),
        )
        img = v.visualize()
        cv2.imshow("img", img)
        cv2.imwrite("imgggggggggggggggg.png",img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
