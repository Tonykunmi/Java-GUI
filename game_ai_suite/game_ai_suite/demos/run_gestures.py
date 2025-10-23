import cv2
from game_ai_suite.cv import GestureTracker


def main():
    tracker = GestureTracker()
    try:
        while True:
            frame = tracker.read()
            if frame is None:
                break
            frame, gesture = tracker.process(frame)
            if gesture:
                x, y = gesture.hand_pos
                cv2.putText(
                    frame,
                    f"{gesture.name}:{gesture.strength:.2f}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
