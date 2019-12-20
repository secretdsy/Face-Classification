import os
import shutil
import random

data_dir = './data/notaug/train'
val_dir = './data/notaug/val/'
ref = 1

for k in range(1,7):
    root_dir = os.path.join(data_dir, str(k))
    output_dir = os.path.join(val_dir, str(k))
    print(len(os.listdir(root_dir)))
    print(len(os.listdir(output_dir)))
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
        else:
            for i in range(len(files)):
                track_list = root
                file_in_track = files[i]
                file_to_copy = track_list + '/' + file_in_track
                if os.path.isfile(file_to_copy) == True:
                    shutil.move(file_to_copy,output_dir)
    print(len(os.listdir(root_dir)))
    print(len(os.listdir(output_dir)))