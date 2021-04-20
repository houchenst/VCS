from comet_ml import Experiment
from data import DeepFashionDataset
from model import VariationalTransformer
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set hyperparameters
hyperparameters = {
    "epochs": 2,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "max_attrs": 21, #similar to sequence length, max number of attributes to describe an outfit
    "embedding_size": 256,
    "latent_size": 256, #size of the learned latent space
    "beta": 1e0, #balances KL and reconstruction term in loss function
    # beta:
    # e1-10 - 1e-4
    "recon_only": 0,
    "num_imgs": 8, #number of images to generate after each epoch
}

# evidence lower bound loss for VAEs
# https://arxiv.org/pdf/1312.6114.pdf
def elbo_loss(recon_y, y, mu, logvar, recon_only=False):
    '''
    recon_y  - our network output, an image reconstructed from the input attributes
    y        - the ground truth image from our dataset
    mu       - our conditional distribution mean in the latent space
    logvar   - the log variance of our conditional distribution in latent space
    '''
    # reconstruction loss
    mse = F.mse_loss(recon_y, y)

    # KL Loss
    kll = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * hyperparameters["beta"]

    return mse + kll, mse, kll

def train_epoch(model, train_loader, optimizer, experiment, epoch, step, recon_only=False):
    model.train()
    with experiment.train():
        epoch_loss = 0
        for img, attrs, mask in tqdm(train_loader):
            img = img.to(device)
            attrs = attrs.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            recon_y, mu, logvar = model(attrs, mask)
            if recon_only:
                loss = F.mse_loss(recon_y,img)
            else:
                loss, mse, kll = elbo_loss(recon_y, img, mu, logvar)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if not recon_only:
                experiment.log_metric("Reconstruction Loss", float(mse.item()), step=step)
                experiment.log_metric("KL Loss", float(kll.item()), step=step)
            experiment.log_metric("loss", float(loss), step=step)
            step+=1
        print(f"\tAverage train loss: {epoch_loss/len(train_loader)}")
    return step

def train(model, train_loader, val_loader, optimizer, experiment, attr_list, val_every=5):
    step=0
    for e in range(hyperparameters["epochs"]):
        print(f"----------    EPOCH {e}    ----------")
        step = train_epoch(model, train_loader, optimizer, experiment, e, step, recon_only=(e<hyperparameters["recon_only"]))
        if e%val_every == 0:
            test(model, val_loader, experiment)
        generate_image(model, val_loader, experiment, attr_list, num_imgs=hyperparameters["num_imgs"], epoch=e)

def test(model, test_loader, experiment, val=True):
    model.eval()
    with experiment.test(), torch.no_grad():
        test_loss=0
        for img, attrs, mask in tqdm(test_loader):
            img = img.to(device)
            attrs = attrs.to(device)
            mask = mask.to(device)
            recon_y, mu, logvar = model(attrs, mask)
            loss, _, _= elbo_loss(recon_y, img, mu, logvar)
            test_loss += loss.item()
        if not val:
            print("TESTING:")
            print(f"\tAverage test loss: {test_loss/len(test_loader)}")
            experiment.log_metric("loss", test_loss)
        else:
            print(f"\tAverage validation loss: {test_loss/len(test_loader)}")

def generate_image(model, val_loader, experiment, attr_list, num_imgs=1, epoch=None, log=True, show=False, save=False):
    '''
    Generate images from clothing description using the model
    '''
    model.eval()
    with torch.no_grad():    
        # i = random.choice(range(hyperparameters["batch_size"]))
        for i, (img, attrs, mask) in enumerate(val_loader):
            if i == num_imgs:
                break
            img = img[:1].to(device)
            attrs = attrs[:1].to(device)
            mask = mask[:1].to(device)
            recon_img, _, _ = model(attrs, mask)
            attr_names = [attr_list[int(attrs[0,i])] for i in range(attrs.shape[1]) if int(attrs[0,i]) != 0]
            img_name = "_".join(attr_names)
            if epoch is not None:
                img_name = f"{epoch+1}_{i+1}_"+img_name
            # print(recon_img[0].shape)
            if log:
                experiment.log_image(recon_img[0].cpu().numpy(), img_name)
            if show:
                plt.imshow()
                plt.show()
            if save:
                # TODO: save image
                pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="DeepFashion dataset root directory")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-g", "--gen", action="store_true", 
                        help="generate an image")
    args = parser.parse_args()

    # Setup comet experiment
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparameters)

    # Load dataset
    train_set = DeepFashionDataset(args.data_root, set_type="train", max_attrs=hyperparameters["max_attrs"])
    val_set = DeepFashionDataset(args.data_root, set_type="val", max_attrs=hyperparameters["max_attrs"])
    test_set = DeepFashionDataset(args.data_root, set_type="test", max_attrs=hyperparameters["max_attrs"])
    attrs = train_set.attrs
    attr2num = train_set.attr2num
    train_loader = DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True)

    # initialize model
    model = VariationalTransformer(hyperparameters["max_attrs"], len(attrs), hyperparameters["embedding_size"], hyperparameters["latent_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    if args.load:
        model.load_state_dict(torch.load('./models/model.pt'))
    if args.train:
        train(model, train_loader, val_loader, optimizer, experiment, attrs)
    if args.test:
        test(model, test_loader, experiment, val=False)
    if args.save:
        torch.save(model.state_dict(), './models/model.pt')
    if args.gen:
        generate_image(model, val_loader, experiment, attrs, log=False, show=True, save=True)

