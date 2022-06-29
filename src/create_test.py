import os
import pandas as pd
from natsort import natsorted

folder_path = "./../data/retinopatia_entrenamiento/"

if __name__ == '__main__':
    os.listdir(folder_path)
    test_img = os.path.join(folder_path, os.listdir(folder_path)[3])
    test_images = os.listdir(test_img)

    images = []
    for i in test_images:
        name = i.split('.')[0]
        images.append(name)
        
    img = natsorted(images)
    df = pd.DataFrame({'image': img})

    df.to_csv('./../data/retinopatia_entrenamiento/dataTest.csv', index=False)
