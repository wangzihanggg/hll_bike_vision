from PIL import Image
import os
from tqdm import tqdm

image_dir_path = r'../dataset/raw_image/'
image_names = os.listdir(image_dir_path)
save_dir = r'../dataset/resize_image/'
resize_size = (640, 480)

image_paths = [os.path.join(image_dir_path, image_name) for image_name in image_names]

if __name__ == '__main__':
    for i, image_path in enumerate(tqdm(image_paths, desc='resize')):
        image = Image.open(image_path)
        image = image.resize(resize_size)
        image.save(save_dir + f'/{i}.jpg')

