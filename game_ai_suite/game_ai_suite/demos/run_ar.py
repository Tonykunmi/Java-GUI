import cv2
from game_ai_suite.ar import Aruco3DTracker


def main():
    tracker = Aruco3DTracker()
    try:
        while True:
            frame = tracker.read()
            if frame is None:
                break
            frame, pose = tracker.process(frame)
            if pose is not None:
                rvec, tvec = pose
                txt = f"t=({tvec[0][0]:.2f},{tvec[0][1]:.2f},{tvec[0][2]:.2f})"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
            cv2.imshow("AR 3D", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
