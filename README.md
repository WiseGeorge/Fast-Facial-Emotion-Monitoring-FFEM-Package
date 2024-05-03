# Fast Facial Emotion Monitoring (FFEM)

[![Downloads](https://static.pepy.tech/badge/ffem)](https://pepy.tech/project/ffem)
[![Downloads](https://static.pepy.tech/badge/ffem/month)](https://pepy.tech/project/ffem)

This package provides a simple and efficient way to perform Facial Emotion Recognition (FER). It leverages advanced techniques and algorithms to detect and classify emotions from facial expressions.

![FFEM](../images/FFEM.gif)

## Main Function

* `MonitorEmotion_From_Video(video_path: str | int, output_path:str) -> None`
  This function takes a video file or a webcam feed as input and performs FER. The results are saved to the specified output path.

## Main Class

* `FaceEmotion_Detection()`
  This class is the backbone of the FER process. It performs face detection and emotion recognition using DeepFace and MediaPipe. The `MonitorEmotion_From_Video` function utilizes this class to carry out its operations.

This package is designed with user-friendliness in mind, making the complex task of FER accessible and straightforward for users. Whether you’re a researcher, a developer, or someone interested in FER, this package can be a valuable tool for your projects.

## FFEM Pipeline

![FFEM](../images/FFEM_Pipeline.png)

## How to Cite:

```
@article{RCCI2806,
	author = {Jorge Martínez Pazos y Arturo Orellana García y William Gómez Fernández y David Batard Lorenzo},
	title = {Monitoring Emotional Response During Mental Health Therapy},
	journal = {Revista Cubana de Ciencias Informáticas},
	volume = {17},
	number = {4},
	year = {2023},
	keywords = {DeepFace, Emotional Response, Face Detection, Facial Emotion Recognition},
	abstract = {Facial emotion recognition is one of the most complex problems in computer vision, due to multiple factors ranging from image brightness to the personality of the individual. This paper built and elucidates the implementation of facial expression recognition solutions, an open-source package called FFEM to easily perform this task, and an application that integrates the previous package, using state-of-the-art models and algorithms for facial detection and emotion recognition mainly coming from MediaPipe and DeepFace with the intention of addressing the challenge of recognizing patients' emotions during cognitive therapy sessions. However, the versatility of this approach allows it to be applied to different industries and tasks, highlighting its potential for diverse use cases.},
	issn = {2227-1899},	url = {https://rcci.uci.cu/?journal=rcci&page=article&op=view&path%5B%5D=2806}
}
```
