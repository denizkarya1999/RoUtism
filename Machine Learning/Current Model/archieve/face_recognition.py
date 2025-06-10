import cv2
from deepface import DeepFace

def analyze_emotions_halfsecond(video_source='experiment.mp4', output_txt='emotions_log.txt'):
    """
    Analyzes emotions in a video only at 0.5-second intervals.
    Logs (0.0, 0.5, 1.0, 1.5, ...) etc. until the video ends.
    
    :param video_source: Path to the video file or integer for webcam (0)
    :param output_txt: Output text file path
    """
    cap = cv2.VideoCapture(video_source)

    # Prepare to write results
    with open(output_txt, 'w') as log_file:
        log_file.write("Timestamp(s),Dominant Emotion\n")

        # Next timestamp we want to analyze (start at 0.0s)
        next_analysis_time = 0.0
        step = 0.5  # half-second intervals

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames or cannot read the video.")
                break

            # Current timestamp in seconds
            frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Check if we've reached or passed the half-second mark
            if frame_timestamp >= next_analysis_time:
                try:
                    # Perform emotion analysis
                    result = DeepFace.analyze(
                        img_path=frame,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    # DeepFace returns a dict if one face, list if multiple
                    # Let's handle only the first or assume single face
                    if isinstance(result, list) and len(result) > 0:
                        # If multiple faces, take the first one (or handle them all if you wish)
                        dominant_emotion = result[0].get('dominant_emotion', 'NoFace')
                    elif isinstance(result, dict):
                        dominant_emotion = result.get('dominant_emotion', 'NoFace')
                    else:
                        dominant_emotion = 'NoFace'

                    # Write to file
                    log_file.write(f"{next_analysis_time:.2f},{dominant_emotion}\n")
                    print(f"Logged: {next_analysis_time:.2f} -> {dominant_emotion}")

                except Exception as e:
                    print(f"Error at {frame_timestamp:.2f}s: {e}")

                # Move to the *next* half-second mark
                next_analysis_time += step

                # Optional: If we've reached the end of the video length, we can break
                # But typically it will break by ret==False at the end.

        cap.release()
    print(f"Analysis complete. Emotions logged to '{output_txt}'")


if __name__ == "__main__":
    analyze_emotions_halfsecond(
        video_source='experiment.mp4',
        output_txt='emotions_log.txt'
    )
