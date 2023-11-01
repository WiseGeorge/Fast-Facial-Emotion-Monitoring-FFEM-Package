# Fast Facial Emotion Monitoring (FFEM)

This package provides a simple and efficient way to perform Facial Emotion Recognition (FER). It leverages advanced techniques and algorithms to detect and classify emotions from facial expressions.

## Main Function

* `MonitorEmotion_From_Video(video_path: str | int, output_path:str) -> None`
  This function takes a video file or a webcam feed as input and performs FER. The results are saved to the specified output path.

## Main Class

* `FaceEmotion_Detection()`
  This class is the backbone of the FER process. It performs face detection and emotion recognition using DeepFace and MediaPipe. The `MonitorEmotion_From_Video` function utilizes this class to carry out its operations.

This package is designed with user-friendliness in mind, making the complex task of FER accessible and straightforward for users. Whether you’re a researcher, a developer, or someone interested in FER, this package can be a valuable tool for your projects.
