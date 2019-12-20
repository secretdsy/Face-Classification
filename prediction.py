import os
import warnings
import numpy as np 
import pandas as pd
import multiprocessing
from pathlib import Path
from keras import backend as K
warnings.filterwarnings(action='ignore')
K.image_data_format()
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

params = {
    # Generator Parameter
    'random_state': 42,

    # Model Parameter
    'img_size': (128, 128),
    'input_shape': (128, 128, 3),
    'batch_size': 48,
#     'nb_workers': multiprocessing.cpu_count() // 2
}

test_dir = Path('./data')
submit_dir = Path('./submit')
save_dir = Path('./savemodel')
label_dir = Path('./label')
train_path = label_dir / 'train_vision.csv'
test_path = label_dir / 'test_vision.csv'

model_filename = 'res152_aug8+real8_val2_ep040_acc-0.9854_vloss-1.7443_vacc-0.9078.h5'
model_filepath = str(save_dir / model_filename)
csv_filename = model_filename + '.csv'

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    str(test_dir),
    classes=['test'],
    target_size=params['img_size'],
    color_mode='rgb',
    class_mode='categorical',
    batch_size=params['batch_size'],
    shuffle=False)

model = load_model(model_filepath)

prediction = model.predict_generator(
    generator=test_generator,
    steps = get_steps(test_generator.samples, params['batch_size']),
    verbose=1,
    workers=params['nb_workers']
)

file_name = test_generator.filenames
for i in range(len(file_name)):
    file_name[i] = file_name[i].split('/')[-1]

predicted_class_indices = np.argmax(prediction, axis=1)

labels = (train_generator.class_indices)
labels = dict((v, k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
submission = pd.read_csv(str(test_path))
submission = pd.DataFrame(submission)

df = pd.DataFrame(file_name, columns=['filename']) 
df2 = pd.DataFrame(predictions, columns=['prediction'])
df['prediction'] = df2

submission = pd.merge(submission, df, on="filename")

submission = submission[['prediction']]
submission.to_csv(str(submit_dir / csv_filename), index=False)
submission.tail()