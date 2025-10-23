from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class GestureEvent:
    name: str
    strength: float
    hand_pos: Tuple[int, int]


class GestureTracker:
    def __init__(self, camera_index: int = 0, min_detection_confidence: float = 0.5) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils
        self.drawer_styles = mp.solutions.drawing_styles

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[GestureEvent]]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        gesture: Optional[GestureEvent] = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Use wrist (0) landmark as coarse hand position
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)

                # Simple gesture heuristic: pinch detection
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                dist = np.hypot(
                    (thumb_tip.x - index_tip.x) * w,
                    (thumb_tip.y - index_tip.y) * h,
                )
                strength = float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))
                name = "pinch" if strength > 0.5 else "open"
                gesture = GestureEvent(name=name, strength=strength, hand_pos=(x, y))

                self.drawer.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.drawer_styles.get_default_hand_landmarks_style(),
                    self.drawer_styles.get_default_hand_connections_style(),
                )
        return frame, gesture

    def release(self) -> None:
        self.cap.release()
        self.hands.close()
