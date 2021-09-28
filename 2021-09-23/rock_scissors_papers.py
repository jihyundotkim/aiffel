from PIL import Image
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import os

tf.random.set_seed(42)

def resize_images(img_path):
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")

# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/training/scissor"
# resize_images(image_dir_path)
# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/training/rock"
# resize_images(image_dir_path)
# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/training/paper"
# resize_images(image_dir_path)
# print("이미지 resize 완료!")

def load_data(img_path, number_of_data=300):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    print("이미지 수:", idx)
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0


print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(300, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

print('Model에 추가된 Layer 개수: ', len(model.layers))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train_norm, y_train, epochs=10)

# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/test/scissor"
# resize_images(image_dir_path)
# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/test/rock"
# resize_images(image_dir_path)
# image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/test/paper"
# resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/AIFFEL/2021-09-23/test"
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이로 정규화

model.evaluate(x_test_norm, y_test)

predicted_result = model.predict(x_test_norm)  # model이 추론한 확률. 
predicted_labels = np.argmax(predicted_result, axis=1)

for i in range(300):
    print (i, "컴퓨터의 예상: ", predicted_labels[i], "정답: ", y_test[i])