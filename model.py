from torch import nn
from torch.nn import functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VariationalTransformer(nn.Module):
    def __init__(self, max_attrs, vocab_size, embedding_size, z_dim, h=8, n=3):
        '''
        Implements a variational transformer encoder, and a deconvolution decoder
        '''
        super().__init__()

        # model hyperparameters
        self.max_attrs = max_attrs
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_heads = h
        self.transformer_layers = n
        self.z_dim = z_dim

        # initialize encoder parameters
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.transformer_layers)
        self.mu_layer = nn.Linear(self.embedding_size, self.z_dim)
        self.logvar_layer = nn.Linear(self.embedding_size, self.z_dim)

        # initialize decoder parameters
        self.initial_decoder_layer = nn.Linear(self.z_dim, 5*5*128)
        # (32,5,5) --> (16,25,25)
        self.convt1 = nn.ConvTranspose2d(128, 64, 5, stride=5)
        self.convt2 = nn.ConvTranspose2d(64,32, 5, stride=1, padding=2)
        # (16,25,25) --> (8,75,75)
        self.convt3 = nn.ConvTranspose2d(32, 16, 7, stride=3, padding=2)
        self.convt4 = nn.ConvTranspose2d(16,8,5,stride=1, padding=2)
        # (8,75,75) --> (4,150,150)
        self.convt5 = nn.ConvTranspose2d(8,4,4,stride=2, padding=1)
        # (4,150,150) --> (3,300,300) 3 channel 300x300 image
        self.convt6 = nn.ConvTranspose2d(4,3,4,stride=2, padding=1)
        self.convt7 = nn.ConvTranspose2d(3,3,9,stride=1, padding=4)

        # parameter used to search latent space via gradient descent
        # this helps us find the attributes that describe an image
        self.z_vec = nn.Parameter(torch.tensor((1,self.z_dim), dtype=torch.float32))

    def reset_z_vec(self):
        '''
        Reset z vector to 0s without assigning new parameter 
        '''
        self.z_vec -= self.z_vec
    
    def encode(self, x, mask):
        # batch_size = x.shape[0]

        # get embeddings
        emb = self.word_embeddings(x)

        # pass through the encoder to mu and log variance
        enc_out = self.transformer_encoder(emb.permute(1,0,2), src_key_padding_mask=mask).permute(1,0,2)
        # average the word embeddings in each sentence
        enc_out = torch.sum(enc_out, axis=1)
        # normalize by number of words per sentence
        words_per_sentence = torch.reshape(torch.sum(torch.logical_not(mask).float(),axis=1), (-1,1))
        enc_out = enc_out / words_per_sentence
        mu = self.mu_layer(enc_out)
        logvar = self.logvar_layer(enc_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        devs = torch.randn_like(std)
        return mu + devs*std

    def decode(self, z):
        y = F.relu(self.initial_decoder_layer(z))
        y = torch.reshape(y, (-1,128,5,5))
        # print(f"Original: {y.shape}")
        y = F.relu(self.convt1(y))
        # print(f"Convt1: {y.shape}")
        y = F.relu(self.convt2(y))
        # print(f"Convt2: {y.shape}")
        y = F.relu(self.convt3(y))
        # print(f"Convt3: {y.shape}")
        y = F.relu(self.convt4(y))
        # print(f"Convt4: {y.shape}")
        y = F.relu(self.convt5(y))
        # print(f"Convt5: {y.shape}")
        y = torch.sigmoid(self.convt6(y))
        # print(f"Convt6: {y.shape}")
        # (batch, 3, 300, 300) --> (batch, 300, 300, 3)
        y = torch.transpose(y, 1, 3)
        return y

    def forward(self, x, mask):
        # encoder
        mu,logvar = self.encode(x,mask)

        # random sampling
        z = self.reparameterize(mu,logvar)

        # decode latent variables
        y = self.decode(z)

        return y,mu,logvar,z


class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim

        # convolutional layers
        # (300,300,3) --> (150,150,8)
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        # (150,150,8) --> (75,75,16)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        # (75,75,16) --> (25,25,32)
        self.conv3 = nn.Conv2d(16, 32, 7, stride=3, padding=3)
        # (25,25,32) --> (5,5,64)
        self.conv4 = nn.Conv2d(32, 64, 9, stride=5, padding=4)
        # convert to z_dim vector
        self.final_img_layer = nn.Linear(5*5*64, self.z_dim)

        # combined z and image layers
        self.dense1 = nn.Linear(2*self.z_dim, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32,1)

    def forward(self, z, img):
        # pass image through convolutional layers
        #(batch, 300, 300, 3) --> (batch, 3, 300, 300)
        img_out = img.transpose(1,3)
        img_out = F.relu(self.conv1(img_out))
        img_out = F.relu(self.conv2(img_out))
        img_out = F.relu(self.conv3(img_out))
        img_out = F.relu(self.conv4(img_out))
        img_out = torch.reshape(img_out, (-1, 5*5*64))
        img_out = F.relu(self.final_img_layer(img_out))

        # combine z and image features
        joint_vec = torch.cat([img_out, z], dim=1)
        joint_vec = F.relu(self.dense1(joint_vec))
        joint_vec = F.relu(self.dense2(joint_vec))
        
        return torch.sigmoid(self.dense3(joint_vec))