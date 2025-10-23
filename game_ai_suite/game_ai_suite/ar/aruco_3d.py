from typing import Optional, Tuple
import cv2
import numpy as np


class Aruco3DTracker:
    def __init__(self, camera_index: int = 0, marker_length_m: float = 0.05) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.marker_length = marker_length_m
        # Use default camera matrix (approx) if not calibrated
        self.camera_matrix = np.array(
            [[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        corners, ids, _ = self.detector.detectMarkers(frame)
        pose = None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                pose = (rvec, tvec)
        return frame, pose

    def release(self) -> None:
        self.cap.release()
