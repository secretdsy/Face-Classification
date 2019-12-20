from pathlib import Path
import os
import cv2
import numpy as np

Extension=".png"
data_dir = Path('./data/aug')
raw_dir = data_dir / 'train'
aug_dir = data_dir / 'train'
dir_list = os.listdir(str(raw_dir))

# aug1
# aug_data --> newaug_c7p3r2sh1z1_adadelta_ep034_vloss-0.1039_vacc-0.9665.h5
# score 89.5, acc: 0.9274

def add_light(image_file):
    image=cv2.imread(image_file)
    gamma = np.random.randint(7,20) / 10 #(0.7, 2.0)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

def sharpen_image(image_file):
    image=cv2.imread(image_file)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

def gausian_blur(image_file, blur):
    image=cv2.imread(image_file)
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

def saturation_image(image_file):
    image=cv2.imread(image_file)
    saturation = np.random.randint(0, 50)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)
    
def salt_image(image_file):
    image=cv2.imread(image_file)
    p = 0.5
    a = np.random.rand() / 50
    noisy = image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

def rotate_image(image_file):
    image=cv2.imread(image_file)
    deg = np.random.uniform(-30,30)
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

def flip_image(image_file):
    image=cv2.imread(image_file)
    dir = 1
    image = cv2.flip(image, dir)
    cv2.imwrite(str(Folder_name) + '/' + file_no + str(i) + Extension, image)

# data augmentation
# 4400 // 2580
for dirname in dir_list:
    LOAD_PATH = raw_dir / dirname
    Folder_name = aug_dir / dirname
    file_list = os.listdir(str(LOAD_PATH))
    cnt = 4400 // len(file_list)
    print('dir:', dirname)
    for i in range(0, cnt):
        print(i)
        for filename in file_list:
            image_file = str(Folder_name) + '/' + filename
            file_no = filename[5:-4] + '_'    
            
            add_light(image_file)
            image_file = str(Folder_name) + '/' + file_no + str(i) + Extension
            rotate_image(image_file)
            if np.random.rand() < 0.5:
                flip_image(image_file)
            if np.random.rand() < 0.2:
                salt_image(image_file)
            if np.random.rand() < 0.3:
                blur = np.random.rand()
                gausian_blur(image_file, blur)
            if np.random.rand() < 0.3:
                saturation_image(image_file)
            if np.random.rand() < 0.3:
                sharpen_image(image_file)