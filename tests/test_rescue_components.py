import struct
import unittest

import cv2
import numpy as np

from exploration import FrontierExplorer, LeftWallFollower
from mapping import MazeMap
from pathfinding import AStarPlanner
from utils.lidar import LidarProcessor
from vision.pipeline import (
    build_letter_templates,
    classify_cognitive_target,
    classify_letter_patch,
    is_cognitive_target,
    preprocess_64,
    validate_inference_gate,
)


class FakeLidar:
    def __init__(self, scans, fov=2.0 * np.pi, max_range=3.0):
        self.scans = iter(scans)
        self.fov = fov
        self.max_range = max_range

    def getFov(self):
        return self.fov

    def getMaxRange(self):
        return self.max_range

    def getRangeImage(self):
        return next(self.scans)


class RescueComponentTests(unittest.TestCase):
    def test_lidar_median_filter_suppresses_single_scan_spike(self):
        processor = LidarProcessor(FakeLidar([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.1, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]), smoothing_radius=0)
        processor.update()
        processor.update()
        result = processor.update()
        self.assertEqual(result[1], 1.0)

    def test_lidar_sectors_and_invalid_ranges(self):
        processor = LidarProcessor(FakeLidar([
            [float("inf")] * 8,
            [1.0] * 8,
            [1.0] * 8,
        ]), smoothing_radius=0)
        ranges = processor.update()
        self.assertTrue(all(distance == 3.0 for distance in ranges))
        self.assertEqual(processor.front(), 3.0)

    def test_lidar_detects_a_raised_return_against_a_flat_wall(self):
        count = 360
        ranges = [0.30] * count
        center = round((np.pi / 2.0 + np.pi) * (count - 1) / (2.0 * np.pi))
        for index in range(center - 4, center + 5):
            ranges[index] = 0.25
        processor = LidarProcessor(
            FakeLidar([ranges]),
            history_size=1,
            smoothing_radius=2,
        )
        processor.update()
        self.assertTrue(processor.has_protrusion(np.pi / 2.0))

    def test_map_uses_quarter_tile_resolution_and_serializes_codes(self):
        maze = MazeMap(tile_size_m=0.12)
        self.assertAlmostEqual(maze.quarter_tile_resolution, 0.03)
        maze.mark((0, 0), "start", visited=True)
        maze.mark((1, 0), "wall")
        maze.mark_token(0.06, 0.0, "H")
        maze.mark_token(0.06, 0.004, "S")
        packet = maze.encode_submission()
        rows, columns = struct.unpack("2i", packet[:8])
        values = packet[8:].decode("ascii").split(",")
        self.assertEqual(len(values), rows * columns)
        self.assertIn("5", values)
        self.assertIn("1", values)
        self.assertIn("SH", values)

    def test_map_supports_area_four_and_passage_codes(self):
        maze = MazeMap()
        maze.mark((0, 0), "area4")
        maze.mark((1, 0), "passage_12")
        maze.mark((2, 0), "hole")
        maze.mark((3, 0), "swamp")
        maze.mark((4, 0), "checkpoint")
        maze.mark((5, 0), "obstacle")
        values = [value for row in maze.matrix() for value in row]
        self.assertIn("*", values)
        self.assertIn("b", values)
        self.assertTrue({"2", "3", "4", "x"}.issubset(values))

    def test_map_rejects_invalid_token_code(self):
        maze = MazeMap()
        with self.assertRaises(ValueError):
            maze.mark_token(0.0, 0.0, "Z")

    def test_astar_routes_around_wall(self):
        cells = {
            (0, 0): {"kind": "start"},
            (1, 0): {"kind": "wall"},
            (0, 1): {"kind": "free"},
            (1, 1): {"kind": "free"},
            (2, 1): {"kind": "free"},
            (2, 0): {"kind": "free"},
        }
        path = AStarPlanner(cells, 0.03).plan((0, 0), (2, 0))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertNotIn((1, 0), path)

    def test_explorer_uses_left_opening_and_avoids_visited_branch(self):
        follower = LeftWallFollower(max_speed=6.28)
        command = follower.command(
            front=1.0,
            left=1.0,
            right=1.0,
            rear_left=1.0,
            current_cell=(0, 0),
            visited=set(),
            yaw=0.0,
            now=0.0,
        )
        self.assertLess(command[0], command[1])
        command = follower.command(
            front=1.0,
            left=1.0,
            right=1.0,
            rear_left=1.0,
            current_cell=(0, 0),
            visited={(0, 1)},
            yaw=0.0,
            now=0.032,
        )
        self.assertGreater(command[0], 0.0)
        self.assertGreater(command[1], 0.0)

    def test_frontier_planner_routes_to_unexplored_free_boundary(self):
        cells = {
            (0, 0): {"kind": "start", "visited": True},
            (1, 0): {"kind": "free"},
        }
        explorer = FrontierExplorer(cells, resolution=0.03)
        path = explorer.choose_path((0, 0), {}, set(), now=0.0)
        self.assertTrue(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (1, 0))
        self.assertEqual(
            explorer.choose_path((0, 0), {}, {path[-1]}, now=0.0),
            [],
        )

    def test_vision_gate_rejects_motion_and_unaligned_pose(self):
        with self.assertRaises(RuntimeError):
            validate_inference_gate(stopped=False, aligned=True)
        with self.assertRaises(RuntimeError):
            validate_inference_gate(stopped=True, aligned=False)
        validate_inference_gate(stopped=True, aligned=True)

    def test_hazmat_classifier_sums_all_five_rings(self):
        colors = {
            "black": (0, 0, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
        }
        cases = (
            (("black", "red", "yellow", "green", "blue"), "F"),
            (("red", "red", "yellow", "green", "blue"), "P"),
            (("red", "yellow", "yellow", "green", "blue"), "C"),
            (("yellow", "yellow", "yellow", "green", "blue"), "O"),
            (("blue", "blue", "blue", "blue", "blue"), None),
        )
        for ring_colors, expected in cases:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            center = (64, 64)
            for radius, name in zip((50, 40, 30, 20, 10), reversed(ring_colors)):
                cv2.circle(image, center, radius, colors[name], thickness=-1)
            with self.subTest(expected=expected, rings=ring_colors):
                self.assertEqual(classify_cognitive_target(image), expected)
                if expected is None:
                    self.assertTrue(is_cognitive_target(image))

    def test_preprocessor_outputs_normalized_64_square(self):
        frame = np.full((24, 40, 3), 255, dtype=np.uint8)
        result = preprocess_64(frame)
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_letter_templates_cover_latin_and_greek_tokens(self):
        templates = build_letter_templates()
        for label, variants in templates.items():
            for template in variants:
                patch = (template * 255.0).astype(np.uint8)
                self.assertEqual(
                    classify_letter_patch(patch, templates)["best_label"],
                    label,
                )


if __name__ == "__main__":
    unittest.main()
