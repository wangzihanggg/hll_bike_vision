from PIL import Image
import cv2


def resize_image_keep_scale(image, size=(512, 512)):
    max_size = max(image.size)
    mask = Image.new('RGB', (max_size, max_size))
    mask.paste(image, (0, 0))
    mask = mask.resize(size)
    return mask


def resize_image_fixed(image, size=(320, 240)):
    mask = cv2.resize(image, size)
    return mask

# def resize_image_fixed(image, size=(320, 240)):
#     image = image.copy()
#     mask = image.resize(size)
#     return mask

# if __name__ == '__main__':
#     image = Image.open('../dataset/raw_image/1.jpg')
#     # image = resize_image_keep_scale(image)
#     image = image.resize((640, 480))
#     print(image.getbands())
#     image.show()
