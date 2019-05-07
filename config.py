# const for captcha generation
from os.path import join
DES_PATH = "gdrive/My Drive/projects/captcha_recognition"
RESULT_FILE_NAME = 'captcha_img'

CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CAP_LEN = 4

NO_TRAIN_CAP = 100000
NO_TEST_CAP = 10000

#config for model training
TRAIN_DIR = join(DES_PATH, RESULT_FILE_NAME + '_train')
TEST_DIR = join(DES_PATH, RESULT_FILE_NAME + '_test')

MODEL_FILENAME = "model.pt"
EPOCHS = 10
LOG_INTERVAL = 10
SEED = 1
LR = 1e-3
BATCH_SIZE = 128
MODEL_NAME = "Model1"
