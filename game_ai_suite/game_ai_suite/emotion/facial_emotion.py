from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class EmotionResult:
    label: str
    confidence: float


class EmotionRecognizer:
    """
    Lightweight face detection (Haar cascade) + placeholder emotion classification
    via simple heuristics (mouth aspect ratio). For production, replace classifier
    with a trained CNN or a transformer.
    """

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def analyze(self, frame: np.ndarray) -> Optional[EmotionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        roi_gray = gray[y : y + h, x : x + w]
        smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return EmotionResult(label="happy", confidence=0.8)
        # Fallback simple heuristic: brightness indicates neutral vs sad
        mean_val = float(np.mean(roi_gray))
        if mean_val < 80:
            return EmotionResult(label="sad", confidence=0.6)
        return EmotionResult(label="neutral", confidence=0.6)
