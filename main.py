# import models
import models
from config import CAP_LEN, TRAIN_DIR, TEST_DIR, MODEL_FILENAME, MODEL_FILENAME, EPOCHS, LOG_INTERVAL, SEED, LR, BATCH_SIZE, MODEL_NAME
from utils import train, test, get_target_from_indices, get_preds_from_output, get_transformation
from os.path import join

from torchvision import datasets, transforms
import torch
import torch.optim as optim

def get_model():
    if not MODEL_NAME in models.__all__:
        raise Except("MODEL_NAME does not exists")
    else:
        return models.__dict__[MODEL_NAME]


def main():
    # Set the seed so results can be reproduced
    torch.manual_seed(SEED)
    # Check if CUDA is available, use it if so
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Model = get_model()
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    transforms = get_transformation(model.img_width, model.img_height)

    # Set up image folder and loader for training and testing
    train_captcha_folder = datasets.ImageFolder(
        TRAIN_DIR, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_captcha_folder,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=1)
    test_captcha_folder = datasets.ImageFolder(
        TEST_DIR, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_captcha_folder,
                                              batch_size=1000,
                                              shuffle=True,
                                              num_workers=1)

    print("Going to train for {} epochs".format(EPOCHS))
    for epoch in range(1, EPOCHS + 1):        
        train(LOG_INTERVAL, model, device, train_loader, optimizer, epoch,
              get_target_from_indices, train_captcha_folder, MODEL_FILENAME)
        test(model, device, test_loader,
             get_target_from_indices, test_captcha_folder)


if __name__ == "__main__":
    main()
