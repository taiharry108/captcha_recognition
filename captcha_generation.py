from captcha.image import ImageCaptcha
from shutil import copyfile
from os.path import join
import numpy as np
import os
from config import CAP_LEN, CHARACTERS, RESULT_FILE_NAME, NO_TRAIN_CAP, NO_TEST_CAP, DES_PATH, OUT_FILENAME
from collections import defaultdict
import pandas as pd


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

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    str_gen = gen_str(size, characters)    

    img_dict = defaultdict(int)
    for i in range(no_of_img):
        text = next(str_gen)
        if i % 10000 == 0:
            print("Generating image {} of {}".format(i, no_of_img))
        img_name = f'{text}_{img_dict[text]}.png'
        img_dict[text] += 1
        captcha.write(text, join(out_dir, img_name))
    return True


def generate_feather(train_dir, valid_dir, out_filename):
    '''
    A function to generate a feather file with the input files and labels

    Args:
        train_dir (str): A directory containing training images
        valid_dir (str): A directory contarining images for validation        

    Returns;
        bool: True for success, False otherwise.
    '''

    def process_filename(filename):
        text = filename.split('_')[0]
        return list(map(lambda x: f'{x[1]}{x[0]}', enumerate(text)))


    rows = []
    for filename in os.listdir(train_dir):
        rows.append((os.path.join('train', filename),
                    process_filename(filename), False))

    for filename in os.listdir(valid_dir):
        rows.append((os.path.join('valid', filename),
                    process_filename(filename), True))

    df = pd.DataFrame(rows, columns=['name', 'label', 'is_valid'])
    df.to_feather(out_filename, index=False)



def main():
    ic = ImageCaptcha()

    train_dir = join(DES_PATH, RESULT_FILE_NAME, 'train')
    valid_dir = join(DES_PATH, RESULT_FILE_NAME, 'valid')

    directories = [train_dir, valid_dir]
    nos_of_img = [NO_TRAIN_CAP, NO_TEST_CAP]

    for directory, no_of_img in zip(directories, nos_of_img):
        generate_captcha(captcha=ic, out_dir=directory,
                         no_of_img=no_of_img, size=CAP_LEN, characters=CHARACTERS)
    

if __name__ == "__main__":
    main()
