# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.vlm.prompt_generator import PromptEngineer
import pdb
from vlfm.utils.habitat_visualizer import HabitatVis
from vlfm.utils.vram_tracker import GPUMemoryTracker
PROMPT_SEPARATOR = "|"

class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._prompt_engineer = PromptEngineer()  # Initialize the dynamic prompt generator
        # self._vlm_client = VLMModelClient(port=int(os.environ.get("LLAVA_PORT", "12182"))) 
        self._text_prompt = self._prompt_engineer.generate_prompt()  #Initialize with the first dynamic prompt
        self._habitat_vis = HabitatVis()
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        # self._tracker = GPUMemoryTracker()  # Initialize the memory tracker
        self._found_ooi = "Unknown"  # Initialize found_ooi

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action

        # Conditionally propose additional frontiers based on found_ooi
        if self._found_ooi == "Yes":  # Only propose additional frontiers if the object is found
            frontiers = self._propose_additional_frontiers(frontiers)

        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _propose_additional_frontiers(self, frontiers: np.ndarray) -> np.ndarray:
        """
        Proposes additional frontier points based on the value map.
        Args:
            frontiers: Existing frontier points.
        Returns:
            Updated list of frontier points.
        """

        # Check if self._value_map is initialized and has the correct shape
        if not hasattr(self, '_value_map') or self._value_map is None:
            print("Value map is not initialized. Returning existing frontiers.")
            return frontiers

        # Access the value map data
        value_map_data = self._value_map._value_map  # Shape: (size, size, value_channels)

        # Reduce the value map to a 2D array by taking the maximum value across channels
        reduced_value_map = np.max(value_map_data, axis=2)  # Shape: (size, size)

        # Threshold for high-value regions
        high_value_threshold = 0.8  # Adjust this threshold as needed

        # Find high-value regions in the value map
        high_value_indices = np.argwhere(reduced_value_map > high_value_threshold)

        if len(high_value_indices) == 0:
            return frontiers  # No high-value regions found

        # Convert indices to world coordinates
        high_value_points = self._indices_to_world(
            high_value_indices,
            self._value_map._episode_pixel_origin,  # Pass episode_pixel_origin
            self._value_map.pixels_per_meter,      # Pass pixels_per_meter
        )

        # Cluster high-value points to avoid adding too many points
        from sklearn.cluster import KMeans
        max_additional_points = 5  # Maximum number of additional points to add
        if len(high_value_points) > max_additional_points:
            kmeans = KMeans(n_clusters=max_additional_points, random_state=0).fit(high_value_points)
            high_value_points = kmeans.cluster_centers_  # Use cluster centers as the proposed points

        # Add new points to the existing frontiers
        updated_frontiers = np.vstack([frontiers, high_value_points])

        return updated_frontiers

    def _indices_to_world(self, indices: np.ndarray, episode_pixel_origin: np.ndarray, pixels_per_meter: float) -> np.ndarray:
        """
        Converts map indices to world coordinates.
        Args:
            indices: Array of indices in the map (shape: (N, 2)).
            episode_pixel_origin: The origin of the episode in pixel coordinates.
            pixels_per_meter: The resolution of the map in pixels per meter.
        Returns:
            Array of world coordinates (shape: (N, 2)).
        """
        # Convert indices to world coordinates
        world_points = (indices - episode_pixel_origin) / pixels_per_meter
        world_points[:, 1] *= -1  # Flip y-axis to match world coordinates
        return world_points

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info


    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        action_scores_list = []
        
        # cv2.imwrite("_obstaclemappppp.png", self._observations_cache["obstacle_map"])

        # # Iterate through all_rgb and run VLM only once every 3 frames
        # for idx, rgb in enumerate(all_rgb):
        #     if idx % 3 == 0:  # Run VLM logic for every 3rd frame
        #         # Get the model's response and action scores
        #         response, action_scores, found_ooi = self._prompt_engineer.process_image_and_prompt(
        #             image1=rgb,
        #             prompt=self._text_prompt,
        #             target_object=self._target_object,
        #         )
        #         action_scores_list.append(action_scores)
        #         self._found_ooi = found_ooi  # Store found_ooi
        #     else:
        #         # Output zero scores for non-trigger frames
        #         zero_scores = {
        #             "Go forward": 0.0,
        #             "Go backward": 0.0,
        #             "Turn right": 0.0,
        #             "Turn left": 0.0,
        #         }
        #         action_scores_list.append(zero_scores)

        for rgb in all_rgb:
            # Get the model's response and action scores
            response, action_scores, found_ooi = self._prompt_engineer.process_image_and_prompt(
                image1=rgb,
                image2=self._observations_cache["obstacle_map"],
                prompt=self._text_prompt,
                target_object=self._target_object,
            )
            
            action_scores_list.append(action_scores)
            self._found_ooi = found_ooi  # Store found_ooi

        del self._observations_cache["obstacle_map"]

        # Update the prompt dynamically
        # parsed_response = self._prompt_engineer.parse_response(response)  # Parse the model's response
        
        self._text_prompt = self._prompt_engineer.generate_prompt()  # Generate a new prompt

        for action_scores, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            action_scores_list, self._observations_cache["value_map_rgbd"]
        ):
            # Convert action_scores dictionary to a numpy array
            scores_array = np.array([action_scores.get(action, 0.0) for action in ["Go forward", "Go backward", "Turn right", "Turn left"]])
            print(scores_array)
            self._value_map.update_map(scores_array, depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]