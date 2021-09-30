# 사진에 스티커 붙여보기

![고양이 수염이 없는 사진](./images/Webcam/photo3.jpg)

이런 일반 얼굴 사진에

![고양이 수염이 붙은 사진](./images/Results/photo3.jpg)

위 사진과 같이 사진에 고양이 수염을 붙여볼 것입니다.

## 라이브러리 `import`하기

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import dlib
import re
```

cv2 - 이미지 처리
matplotlib - 진행하면서 이미지를 볼 수 있기 위해

numpy - 이미지를 numpy array 형태로 사용하기 위해

os - 이미지의 path를 지정하기 위해

dlib - 얼굴 인식을 위해

re - 파일명 처리 정규표현식 때문에 (없어도 무방함)

##  add_whiskers 함수 구현하기

```python
def add_whiskers(img_path, sticker_path, detector_hog, landmark_predictor):
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (640, 360))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dlib_rect = detector_hog(img_rgb, 1)[0]
    list_landmarks = []
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)
    x = list_landmarks[0][30][0]
    y = list_landmarks[0][30][1] + dlib_rect.height() // 20
    w = dlib_rect.width()
    h = dlib_rect.height()
    img_sticker = cv2.imread(sticker_path)
    img_sticker = cv2.cvtColor(img_sticker, cv2.COLOR_BGR2RGB)
    img_sticker = cv2.resize(img_sticker, (w,h), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    refined_x = x - w // 2
    refined_y = y - h // 2
    sticker_area = img_rgb[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    img_rgb[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker>=[100,100,100], sticker_area, cv2.addWeighted(img_sticker, 0.5, sticker_area, 0.5, 0))
    plt.imshow(img_rgb)
    cv2.imwrite(os.getenv('HOME')+'/AIFFEL/2021-09-30/images/Results/'+re.search(r'photo[0-9]\.jpg',image_path).group(), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
```
### 얼굴 이미지 불러와서 크기와 컬러스페이스 조정하기
```python
img_bgr = cv2.imread(img_path)
img_bgr = cv2.resize(img_bgr, (640, 360))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### 얼굴 영역과 영역 내 이목구비(landmarks) 인식하기
```python
dlib_rect = detector_hog(img_rgb, 1)[0]
list_landmarks = []
points = landmark_predictor(img_rgb, dlib_rect)
list_points = list(map(lambda p: (p.x, p.y), points.parts()))
list_landmarks.append(list_points)
```
### 스티커 위치 지정하기
```python
x = list_landmarks[0][30][0]
y = list_landmarks[0][30][1] + dlib_rect.height() // 20
w = dlib_rect.width()
h = dlib_rect.height()
refined_x = x - w // 2
refined_y = y - h // 2
```
우선 코의 위치인 30번 랜드마크를 기준으로 `x`와 `y`를 정합니다. `y`의 경우 랜드마크 위치보다 살짝 아래로 내리는 게 더 코 위치에 맞는 것 같아 직사각형 높이의 1/20만큼 내려주었습니다. 그 다음 얼굴 크기에 맞게 너비와 높이를 정해줍니다. `refined_x`와 `refined_y`를 높이/너비의 반을 뺀 값으로 정의하는 이유는 랜드마크 위치가 실제로 스티커가 들어갈 자리의 중앙이며 이를 왼쪽 위로 맞춰주기 위함입니다.

### 스티커 이미지 불러와서 사이즈와 컬러 스페이스 조정하기
```python
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.cvtColor(img_sticker, cv2.COLOR_BGR2RGB)
img_sticker = cv2.resize(img_sticker, (w,h), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
```
`fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA` 부분은 이미지를 줄일 때 화질 손실을 줄이기 위해서 추가해주었습니다. 사진을 줄일 때는 `INTER_AREA`가 낫고 사진 크기를 키울 때는 `INTER_LINEAR`가 더 좋다고 합니다.

### 스티커 영역을 정의해 스티커 붙이기
```python
sticker_area = img_rgb[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_rgb[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker>=[100,100,100], sticker_area, cv2.addWeighted(img_sticker, 0.5, sticker_area, 0.5, 0))
```
스티커가 들어갈 자리의 위쪽 가장자리(`refined_y`)부터 아래쪽 가장자리(`refined_y+img_sticker.shape[0]`)까지, 그리고 왼쪽 가장자리(`refined_x`)부터 오른쪽 가장자리(`refined_x+img_sticker.shape[1]`)까지 스티커 영역을 지정해주고, 스티커 영역 내에서 `img_sticker`의 


## 얼굴 인식기 인스턴스 및 각종 변수 선언

```python
num_of_images = 8
detector_hog = dlib.get_frontal_face_detector()
model_path = os.getenv('HOME')+'/AIFFEL/2021-09-30/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
sticker_path = os.getenv('HOME')+'/AIFFEL/2021-09-30/images/cat-whiskers.png'
for i in range(1, num_of_images+1):
    plt.figure(figsize=(640, 360))
    plt.subplot(1, num_of_images, i)
    image_path = os.getenv('HOME')+'/AIFFEL/2021-09-30/images/Webcam/photo'+str(i)+'.jpg'
    add_whiskers(image_path, sticker_path, detector_hog, landmark_predictor)
```

## 문제점과 해결방안 고민

![얼굴인식이 잘못된 사진](./images/Results/error.png)

얼굴을 한 쪽으로 좀 많이 돌렸더니 잘못된 영역이 인식되는 것을 볼 수 있습니다. [OpenCV 정보 사이트](https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)에 따르면 `dlib`의 `get_frontal_face_detector()` 자체가 frontal이 아닌, 많이 틀어져 있는 얼굴은 인식을 하지 못한다고 합니다. 얼굴의 각도가 달라져도 인식이 잘 되는 라이브러리로는 OpenCV의 DNN Face Detector가 있다고 합니다. 나중에 DNN Face Detector와 OpenCV의 사용법을 더 익혀서 다시 적용해 봐야겠습니다.
