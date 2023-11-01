import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from deepface import DeepFace
import random
import keras

class FaceEmotionDetection():
    """
    This class facilitates Facial Emotion Recognition (FER) by performing face detection and emotion recognition using DeepFace and MediaPipe.

    Attributes:
    bg_color (list): A list of RGB color tuples used for visualization.

    Methods:
    findFaces(img, fancyDraw=False, draw=True, detector_backend='mediapipe', cuadrant=True):
        Detects faces in an image and performs emotion recognition on the detected faces.

    get_dominant_emotion(img, detector_backend='mediapipe'):
        Analyzes an image using DeepFace to determine the dominant emotion.

    draw_emotions(img, emotions, size=400):
        Draws an emotion quadrant chart on an image.
    """

    def __init__(self):
        self.bg_color = [(255, 77, 77),(255, 117, 44),(0, 191, 255),(64, 224, 208)]

    def findFaces(self,img, fancyDraw=False,draw=True,detector_backend='mediapipe',cuadrant=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs = []        
        height, width = img.shape[:2]
    
        # calcula las coordenadas del centro del círculo
        radius = 100
        x = width - radius - 10
        y = height - radius - 10
        center = (x, y)
        result = 'neutral'
        
        if self.imgRGB is not None:
            result, emotions, bbox = self.get_dominant_emotion(self.imgRGB,detector_backend)
            #self.draw_emotion_quadrant(self.imgRGB,emotions,center)
            
            bboxs.append([bbox])
            if draw:
                if fancyDraw:
                    img = self.fancyDraw(img,bbox,self.bg_color[1],30,4,1)
                else:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[3], bbox[2]+bbox[4]), (255,0,255),2)
                self.draw_text_with_background(img,result.title(),bbox,(255, 255, 255),self.bg_color[1])

            if cuadrant:
                self.draw_emotions(img,emotions)

        return img, bboxs, result, emotions

    def get_dominant_emotion(self, img, detector_backend='mediapipe'):
        # analiza la imagen con DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend=detector_backend)
        #print(result)
        emotion = result[0]
        # obtiene el diccionario de emociones
        emotions = emotion['emotion']
        
        # encuentra la emoción con el mayor valor
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_emotion = emotion['dominant_emotion']
        # obtiene las coordenadas del cuadro delimitador de la cara
    
        #print(emotion['region'])
        face_bbox = emotion['region']
        bbox = int(face_bbox['x']), int(face_bbox['y']), int(face_bbox['w']), int(face_bbox['h'])

        print(dominant_emotion)
        print(type(dominant_emotion))
        return dominant_emotion, emotions, bbox
    


    def draw_emotions(self,img, emotions, size=400):
        # Define los colores para cada emoción
        colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 255, 255),
            'fear': (255, 0, 255),
            'happy': (0, 255, 0),
            'sad': (255, 255, 0),
            'surprise': (255, 165, 0),
            'neutral': (255, 0, 0)
        }

        # Define las coordenadas polares para cada emoción
        coords = {
            'angry': (0.5 * np.pi, 0.8 * size / 2),
            'disgust': (1.5 * np.pi, 0.8 * size / 2),
            'fear': (1 * np.pi, 0.8 * size / 2),
            'happy': (1.5 * np.pi, 0.8 * size / 2),
            'sad': (1 * np.pi, 0.8 * size / 2),
            'surprise': (2 * np.pi, 0.8 * size / 2),
            'neutral': (0, 0)
        }

        # Dibuja el círculo gris transparente en la esquina inferior derecha de la imagen
        overlay = img.copy()
        center = (img.shape[1] - size //2 -10 , img.shape[0] - size //2 -10)
        cv2.circle(overlay, center , size //2 , (128,128,128), -1)
        alpha = 0.4
        img = cv2.addWeighted(overlay,alpha,img.copy(),1-alpha ,0)

        # Dibuja los cuadrantes para cada emoción
        for emotion, color in colors.items():
            theta, rho = coords[emotion]
            x = int(rho * np.cos(theta) + center[0])
            y = int(rho * np.sin(theta) + center[1])
            cv2.line(img,(center[0],center[1]),(x,y),color ,2)
            cv2.putText(img ,emotion.capitalize(),(x-20,y-20),cv2.FONT_HERSHEY_SIMPLEX ,0.5,color ,2)

        # Calcula el desplazamiento del punto rojo en función de las emociones
        dx = int((emotions['happy'] - emotions['sad']) * size /4)
        dy = int((emotions['angry'] - emotions['fear']) * size /4)

        # Dibuja el punto rojo en el centro del cuadrante
        cv2.circle(img,(center[0]+dx ,center[1]+dy),size//20 ,(148 ,49 ,38),-1)

        return img


    def draw_emotion_quadrant(self, img, emotions, center):
        # radio del círculo
        radius = 100
        
        # grosor de las líneas
        thickness = 2
        
        # colores para las emociones
        emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 0),
            'fear': (255, 0, 255),
            'happy': (0, 255, 255),
            'sad': (255, 0, 0),
            'surprise': (255, 255, 0),
            'neutral': (128, 128, 128)
        }
        
        # dibuja el círculo
        cv2.circle(img, center, radius, (255, 255, 255), thickness)
        
        # dibuja las líneas del cuadrante
        cv2.line(img, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255, 255, 255), thickness)
        cv2.line(img, (center[0], center[1] - radius), (center[0], center[1] + radius), (255, 255, 255), thickness)
        
        # calcula la posición del punto central según los valores de las emociones
        x_offset = int((emotions['happy'] - emotions['sad']) / 100 * radius)
        y_offset = int((emotions['angry'] - emotions['fear']) / 100 * radius)
        
        # dibuja el punto central
        cv2.circle(img, (center[0] + x_offset, center[1] + y_offset), 5, (0, 0, 0), -1)
        
        # muestra las emociones en el cuadrante
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        for emotion, value in emotions.items():
            color = emotion_colors[emotion]
            if emotion == 'angry':
                position = (center[0] - radius + 10, center[1] - radius + 20)
            elif emotion == 'disgust':
                position = (center[0] + radius - 60, center[1] - radius + 20)
            elif emotion == 'fear':
                position = (center[0] + radius - 40 , center[1] + radius -10)
            elif emotion == 'happy':
                position = (center[0] - radius +10 , center[1] + radius -10)
            elif emotion == 'sad':
                position = (center[0] - radius +10 , center[1])
            elif emotion == 'surprise':
                position = (center[0] + radius -70 , center[1])
            elif emotion == 'neutral':
                position = (center[0], center[1])
            
            cv2.putText(img=img,
                        text=f"{emotion}: {value}%",
                        org=position,
                        fontFace=font,
                        fontScale=font_scale,
                        color=color,
                        thickness=thickness)



    def fancyDraw(self, img, bbox, color=(255, 0, 255), l=20, t=4, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Drawing Rectangle
        cv2.rectangle(img, bbox, color, rt)
        # Top Left X,Y
        cv2.line(img, (x,y), (x+l,y), color, t)
        cv2.line(img, (x,y), (x,y+l), color, t)
        # Top Right X,Y
        cv2.line(img, (x1,y), (x1-l,y), color, t)
        cv2.line(img, (x1,y), (x1,y+l), color, t)
        # Buttom Left X,Y
        cv2.line(img, (x,y1), (x+l,y1), color, t)
        cv2.line(img, (x,y1), (x,y1-l), color, t)
        # Buttom Right X,Y
        cv2.line(img, (x1,y1), (x1-l,y1), color, t)
        cv2.line(img, (x1,y1), (x1,y1-l), color, t)

        return img

    
    def draw_text_with_background(self, img, text, bbox, text_color, bg_color):
        text_offset = 10
        padding = 6
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = int(bbox[0])
        text_y = int(bbox[1]) - text_offset

        # Calcular las coordenadas del rectángulo
        rect_x1 = text_x - padding
        rect_y1 = text_y - text_size[1] - padding
        rect_x2 = text_x + text_size[0] + padding
        rect_y2 = text_y + padding

        # Dibujar el rectángulo con bordes redondeados
        radius = 10
        cv2.ellipse(img, (rect_x1 + radius, rect_y1 + radius), (radius, radius), 180, 0, 90, bg_color, -1)
        cv2.ellipse(img, (rect_x2 - radius, rect_y2 - radius), (radius, radius), 0, 0, 90, bg_color, -1)
        cv2.ellipse(img, (rect_x1 + radius, rect_y2 - radius), (radius, radius), 90, 0, 90,bg_color,-1)
        cv2.ellipse(img,(rect_x2-radius ,rect_y1+radius),(radius ,radius) ,270 ,0 ,90,bg_color ,-1)
        cv2.rectangle(img,(rect_x1 ,rect_y1+radius),(rect_x2 ,rect_y2-radius),bg_color ,-1)
        cv2.rectangle(img,(rect_x1+radius ,rect_y1),(rect_x2-radius ,rect_y2),bg_color ,-1)

        # Dibujar el texto encima del rectángulo
        cv2.putText(img,text,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,text_color ,2)

        return img

def MonitorEmotion_From_Video(video_path: str|int, output_path:str)->None:
    """
    This function performs Facial Emotion Recognition (FER) on a given video.

    Parameters:
        video_path (str): The path to the video file. If the value is 0, the function uses the available webcam.
        output_path (str): The path to save the processed video file

    The function takes the input video and applies face detection followed by emotion recognition on the detected faces. It then annotates the image with information about the bounding box and the detected emotion.

    The function operates in a loop until the 'q' key is pressed, processing each frame of the video for face and emotion detection. It also calculates and displays the frames per second (FPS) on the output video for performance analysis.

    The output is the processed video file in the given output_path with annotated faces and emotions.
        - The output should look like: 'path/videoName.avi' or similars
    """

    # Define la ruta del video
    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    pTime = 0
    cTime = 0
    detector = FaceEmotionDetection()
    frame_count = 0

    # Define el codec y crea un objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640,480))

    while True:
        succes, img = cap.read()

        # verifica si se leyó el fotograma correctamente
        if not succes:
            break

        # Redimensiona la imagen a las dimensiones deseadas
        img = cv2.resize(img, (640, 480))  # Reemplaza con las dimensiones deseadas

        # incrementa el contador de fotogramas
        frame_count += 1

        img, bboxes, result, cuadrant = detector.findFaces(img,frame_count,True)

        # Time Management
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

        # Escribe el fotograma en el archivo de salida
        out.write(img)

        cv2.imshow('Image',img)

        # verifica si se presionó la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
