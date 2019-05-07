from captcha.image import ImageCaptcha
from shutil import copyfile
from os.path import join
import numpy as np
import os
from config import CAP_LEN, CHARACTERS, RESULT_FILE_NAME, NO_TRAIN_CAP, NO_TEST_CAP, DES_PATH


def gen_str(size, characters):
    '''
    A generator to generate random string as captcha

    Args:
        size (int): The length of the captcha string
        characters (str): A string with unique characters

    Yields:
        str: Generated captcha
    '''
    while True:
        positions = np.random.randint(len(characters), size=size)
        yield ''.join(map(lambda x: characters[x], positions))


def generate_captcha(captcha, out_dir, no_of_img=10000, size=CAP_LEN, characters=CHARACTERS):
    '''
    A function to generate a number of captcha images and write them to an output directory

    Args:
        captcha (ImageCaptcha): An ImageCaptcha object for the generation of captcha from the module captcha
        out_dir (str): Output directory for the captcha images
        no_of_img (int): Number of images to be generated
        size (int): The length of the captcha string
        characters (str): A string with unique characters

    Returns;
        bool: True for success, False otherwise.
    '''
    str_gen = gen_str(size, characters)
    for i in range(no_of_img):
        text = next(str_gen)
        out_path = join(out_dir, text)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        if i % 10000 == 0:
            print("Generating image {} of {}".format(i, no_of_img))
        img_name = '{}.png'.format(len(os.listdir(out_path)))
        captcha.write(text, join(out_path, img_name))
    return True

def main():
    ic = ImageCaptcha()

    train_dir = join(DES_PATH, RESULT_FILE_NAME + '_train')
    test_dir = join(DES_PATH, RESULT_FILE_NAME + '_test')

    directories = [train_dir, test_dir]
    nos_of_img = [NO_TRAIN_CAP, NO_TEST_CAP]

    for directory, no_of_img in zip(directories, nos_of_img):
        generate_captcha(captcha=ic, out_dir=directory,
                         no_of_img=no_of_img, size=CAP_LEN, characters=CHARACTERS)

if __name__ == "__main__":
    main()
