import cv2
from game_ai_suite.emotion import EmotionRecognizer


def main():
    cap = cv2.VideoCapture(0)
    recog = EmotionRecognizer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = recog.analyze(frame)
            if result:
                cv2.putText(
                    frame,
                    f"{result.label} ({result.confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Emotion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
