import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration --- #
# Root directory for dataset
DATAROOT = "./data"
# Number of workers for dataloader
WORKERS = 2
# Batch size during training
BATCH_SIZE = 128
# Spatial size of training images. All images will be resized to this size
# using a transformer.
IMAGE_SIZE = 64
# Number of channels in the training images. For color images this is 3
CHANNELS = 3
# Size of z latent vector (i.e. size of generator input)
LATENT_VECTOR_SIZE = 100
# Size of feature maps in generator
GENERATOR_FEATURE_MAP_SIZE = 64
# Size of feature maps in discriminator
DISCRIMINATOR_FEATURE_MAP_SIZE = 64
# Number of training epochs
NUM_EPOCHS = 5
# Learning rate for optimizers
LEARNING_RATE = 0.0002
# Beta1 hyperparam for Adam optimizers
BETA1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
NGPU = 1

# Decide which device we want to run on
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

# --- Data Loading --- #
def load_dataset():
    """Loads the CelebA dataset and applies transformations."""
    print("Loading dataset...")
    dataset = dset.ImageFolder(root=DATAROOT,
                               transform=transforms.Compose([
                                   transforms.Resize(IMAGE_SIZE),
                                   transforms.CenterCrop(IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=WORKERS)
    print("Dataset loaded.")
    return dataloader

# --- Weights Initialization --- #
def weights_init(m):
    """Custom weights initialization called on Generator and Discriminator."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- Generator Model --- #
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAP_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAP_SIZE * 8),
            nn.ReLU(True),
            # state size. (GENERATOR_FEATURE_MAP_SIZE*8) x 4 x 4
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAP_SIZE * 8, GENERATOR_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAP_SIZE * 4),
            nn.ReLU(True),
            # state size. (GENERATOR_FEATURE_MAP_SIZE*4) x 8 x 8
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAP_SIZE * 4, GENERATOR_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAP_SIZE * 2),
            nn.ReLU(True),
            # state size. (GENERATOR_FEATURE_MAP_SIZE*2) x 16 x 16
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAP_SIZE * 2, GENERATOR_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAP_SIZE),
            nn.ReLU(True),
            # state size. (GENERATOR_FEATURE_MAP_SIZE) x 32 x 32
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAP_SIZE, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (CHANNELS) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# --- Discriminator Model --- #
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (CHANNELS) x 64 x 64
            nn.Conv2d(CHANNELS, DISCRIMINATOR_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DISCRIMINATOR_FEATURE_MAP_SIZE) x 32 x 32
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAP_SIZE, DISCRIMINATOR_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DISCRIMINATOR_FEATURE_MAP_SIZE*2) x 16 x 16
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 2, DISCRIMINATOR_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DISCRIMINATOR_FEATURE_MAP_SIZE*4) x 8 x 8
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 4, DISCRIMINATOR_FEATURE_MAP_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DISCRIMINATOR_FEATURE_MAP_SIZE*8) x 4 x 4
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAP_SIZE * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# --- Training Function --- #
def train_dcgan():
    """Trains the DCGAN model."""
    dataloader = load_dataset()

    # Initialize Generator and Discriminator
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Print the models
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=DEVICE)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    print("Starting DCGAN training loop...")
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, LATENT_VECTOR_SIZE, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the real and fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    print("DCGAN training finished.")

    # Plot the training losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("dcgan_losses.png")
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("dcgan_generated_images.png")
    plt.show()

if __name__ == "__main__":
    train_dcgan()
