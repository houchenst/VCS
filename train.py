from comet_ml import Experiment
from data import DeepFashionDataset
from model import VariationalTransformer, Discriminator
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
    "epochs": 2000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "max_attrs": 21, #similar to sequence length, max number of attributes to describe an outfit
    "embedding_size": 1024,
    "latent_size": 1024, #size of the learned latent space
    "beta": 1e-5, #balances KL and reconstruction term in loss function
    # beta:
    # e1-10 - 1e-4
    "recon_only": 0, #number of epochs at the start of training that should use reconstruction loss only
    "num_imgs": 8, #number of images to generate after each epoch
    "recon": "both", #'gan' or 'mse' specifies the reconstruction loss to use
}

def gan_loss(true_images, gen_images, disc, zs):
    '''
    Used for reconstruction loss, approximates joint distribution P(z,y) for latent variable z and image y
    '''
    real_term = torch.mean(torch.log(disc(zs, true_images)+1e-8))
    gen_term = torch.mean(torch.log(1. - disc(zs, gen_images)+1e-8))
    return real_term + gen_term

# mse takes disc and zs to match gan_loss signature
def mse_loss(true_images, gen_images, disc, zs):
    return F.mse_loss(true_images, gen_images)

def both_loss(true_images, gen_images, disc, zs):
    return mse_loss(true_images, gen_images, disc, zs) + gan_loss(true_images, gen_images, disc, zs)

# specify reconstruction loss based on hyperparameters
recon_loss = gan_loss if hyperparameters["recon"] == "gan" else (mse_loss if hyperparameters["recon"]=="mse" else both_loss)

# evidence lower bound loss for VAEs
# https://arxiv.org/pdf/1312.6114.pdf
def elbo_loss(recon_y, y, mu, logvar, zs, disc, recon_only=False):
    '''
    recon_y  - our network output, an image reconstructed from the input attributes
    y        - the ground truth image from our dataset
    mu       - our conditional distribution mean in the latent space
    logvar   - the log variance of our conditional distribution in latent space
    '''
    # reconstruction loss
    rl = recon_loss(y, recon_y, disc, zs)

    # KL Loss
    kll = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * hyperparameters["beta"]

    return rl + kll, rl, kll

def train_epoch(model, disc, train_loader, optimizer, disc_optimizer, experiment, epoch, step, recon_only=False):
    model.train()
    with experiment.train():
        epoch_loss = 0
        for img, attrs, mask in tqdm(train_loader):
            img = img.to(device)
            attrs = attrs.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            recon_y, mu, logvar, z = model(attrs, mask)
            if recon_only:
                loss = mse_loss(img,recon_y, disc, z)
            else:
                loss, rl, kll = elbo_loss(recon_y, img, mu, logvar, z, disc)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if hyperparameters["recon"]=="gan" or hyperparameters["recon"]=="both":
                # discriminator is better than generator to start
                if epoch % 2 == 0:
                    disc_optimizer.zero_grad()
                    disc_loss = -1 * gan_loss(img, recon_y.detach(), disc, z.detach())
                    disc_loss.backward()
                    disc_optimizer.step()
                    experiment.log_metric("Discriminator Loss", float(disc_loss.item()), step=step)
            if not recon_only:
                experiment.log_metric(f"{hyperparameters['recon']} loss", float(rl.item()), step=step)
                experiment.log_metric("KL Loss", float(kll.item()), step=step)
            experiment.log_metric("loss", float(loss), step=step)
            step+=1
        print(f"\tAverage train loss: {epoch_loss/len(train_loader)}")
    return step

def train(model, disc, train_loader, val_loader, optimizer, disc_optimizer, experiment, attr_list, save=False, val_every=5):
    step=0
    for e in range(hyperparameters["epochs"]):
        print(f"----------    EPOCH {e}    ----------")
        step = train_epoch(model, disc, train_loader, optimizer, disc_optimizer, experiment, e, step, recon_only=(e<hyperparameters["recon_only"]))
        if save:
            torch.save(model.state_dict(), './models/model.pt')
            torch.save(disc.state_dict(), './models/disc.pt')
        if e%val_every == 0:
            test(model, disc, val_loader, experiment)
        generate_image(model, val_loader, experiment, attr_list, num_imgs=hyperparameters["num_imgs"], epoch=e)

def test(model, disc, test_loader, experiment, val=True):
    model.eval()
    with experiment.test(), torch.no_grad():
        test_loss=0
        for img, attrs, mask in tqdm(test_loader):
            img = img.to(device)
            attrs = attrs.to(device)
            mask = mask.to(device)
            recon_y, mu, logvar,zs = model(attrs, mask)
            loss, _, _= elbo_loss(recon_y, img, mu, logvar, zs, disc)
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
            recon_img, mu, logvar,z = model(attrs, mask)
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
    parser.add_argument("-S", "--smooth", action="store_true",
                        help="generate 3 images to check the smoothness of the latent space")
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
    train_loader = DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    # initialize model
    model = VariationalTransformer(hyperparameters["max_attrs"], len(attrs), hyperparameters["embedding_size"], hyperparameters["latent_size"]).to(device)
    disc = Discriminator(hyperparameters["latent_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    disc_optimizer = optim.Adam(disc.parameters(), lr=hyperparameters["learning_rate"])

    if args.load:
        model.load_state_dict(torch.load('./models/model.pt'))
        disc.load_state_dict(torch.load('./models/disc.pt'))
    if args.train:
        train(model, disc, train_loader, val_loader, optimizer, disc_optimizer, experiment, attrs, save=args.save)
    if args.test:
        test(model, disc, test_loader, experiment,  val=False)
    # save every epoch during training instead 
    # if args.save:
    #     torch.save(model.state_dict(), './models/model.pt')
    #     torch.save(disc.state_dict(), './models/disc.pt')
    if args.gen:
        generate_image(model, val_loader, experiment, attrs, log=False, show=True, save=True)
    if args.smooth:
        model.eval()
        with torch.no_grad():
            z1 = torch.zeros((hyperparameters["latent_size"])).to(device)
            z2 = 0.01*torch.ones((hyperparameters["latent_size"])).to(device)
            z3 = 1.*torch.ones((hyperparameters["latent_size"])).to(device)
            z4 = 10.*torch.ones((hyperparameters["latent_size"])).to(device)
            vecs = torch.vstack([z1,z2,z3,z4]).to(device)
            ys = model.decode(vecs)
            experiment.log_image(ys[0].cpu().numpy(), "z_zero")
            experiment.log_image(ys[1].cpu().numpy(), "z_hundredth")
            experiment.log_image(ys[2].cpu().numpy(), "z_one")
            experiment.log_image(ys[3].cpu().numpy(), "z_ten")


