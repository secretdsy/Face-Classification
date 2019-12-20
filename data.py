import os
import shutil
import glob
import pandas as pd
import random
DATA_PATH = '/Users/doyoung/deep-learning/face'
DIR_PATH = '/Users/doyoung/deep-learning/face/train'

data_dir = '/Users/doyoung/deep-learning/face/train/'
output_dir = '/Users/doyoung/deep-learning/face/val2/'
ref = 1

a=len(os.listdir(data_dir))
b=len(os.listdir(output_dir))
for k in range(1,7):
    root_dir = os.path.join(data_dir, str(k))
    for root, dirs, files in os.walk(root_dir):
        number_of_files = len(os.listdir(root)) 
        if number_of_files > ref:
            ref_copy = int(round(0.2 * number_of_files))
            for i in range(ref_copy):
                chosen_one = random.choice(os.listdir(root))
                file_in_track = root
                file_to_copy = file_in_track + '/' + chosen_one
                if os.path.isfile(file_to_copy) == True:
                    shutil.move(file_to_copy,output_dir)
                    print (file_to_copy)
        else:
            for i in range(len(files)):
                track_list = root
                file_in_track = files[i]
                file_to_copy = track_list + '/' + file_in_track
                if os.path.isfile(file_to_copy) == True:
                    shutil.move(file_to_copy,output_dir)
                    print (file_to_copy)
print('Finished !')
print(a)
print(b)
print(len(os.listdir(root_dir)))
print(len(os.listdir(output_dir)))


# os.mkdir(os.path.join(DATA_PATH, 'test'))
# os.mkdir(os.path.join(DATA_PATH, 'train'))

# print(len(os.listdir(DIR_PATH)))
# for i in range(1,7):
#     os.mkdir(DIR_PATH + '/' + str(i) + '/')
# df_train = pd.read_csv(os.path.join(DATA_PATH, 'train_vision.csv'))
# df_test = pd.read_csv(os.path.join(DATA_PATH, 'test_vision.csv'))
# print(df_train.head)
# print(df_train['filename'].head())

# IMG_PATH = '/Users/doyoung/deep-learning/face/faces_images'
# TRAIN_PATH = '/Users/doyoung/deep-learning/face/train'
# TEST_PATH = '/Users/doyoung/deep-learning/face/test'
# # train data move
# for i in range(len(df_train)):
#     origin_path = os.path.join(IMG_PATH, df_train['filename'][i])
#     new_path = TRAIN_PATH + '/' + str(df_train['label'][i]) + '/' + df_train['filename'][i]
#     shutil.move(origin_path, new_path)
# # test data move
# for i in range(len(df_test)):
#     origin_path = os.path.join(IMG_PATH, df_test['filename'][i])
#     new_path = os.path.join(TEST_PATH, df_test['filename'][i])
#     shutil.move(origin_path, new_path)

# #각 폴더 개수세기, train = [1723, 601, 230, 2568, 374, 354] = 5850, test = 2000
# for i in range(1,7):
#     # img_cnt = len(glob.glob1(os.path.join(TRAIN_PATH, str(i)), '*.png'))
#     img_cnt = len(glob.glob1(TEST_PATH, '*.png'))
# for i in range(1,7):
#     print(len(os.listdir(os.path.join(TRAIN_PATH, str(i)))))

# print(len(os.listdir(os.path.join(DATA_PATH, 'test'))))
# print(os.listdir(os.path.join(DATA_PATH, 'train')))