import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import argparse

# --- Configuration --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 350
LOAD_MODEL = False
CONTENT_PATH = "content.png"
STYLE_PATH = "style.png"
GENERATED_PATH = "generated.png"

# --- Image Loading and Preprocessing --- #
def load_image(image_name):
    """Loads an image and applies transformations."""
    image = Image.open(image_name)
    loader = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Normalization is typically handled by the VGG model internally or not strictly necessary for style transfer
    ])
    image = loader(image).unsqueeze(0) # Add batch dimension
    return image.to(DEVICE)

# --- VGG Model for Feature Extraction --- #
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"] # Corresponds to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        self.model = models.vgg19(pretrained=True).features[:29] # Take up to relu5_1

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# --- Training Function --- #
def train_neural_style_transfer(content_img_path, style_img_path, generated_img_path):
    """Performs neural style transfer."""
    print("Loading content and style images...")
    original_img = load_image(content_img_path)
    style_img = load_image(style_img_path)

    # Initialize generated image with content image or random noise
    generated = original_img.clone().requires_grad_(True)
    # generated = torch.randn(original_img.shape, device=DEVICE, requires_grad=True)

    model = VGG().to(DEVICE).eval()
    optimizer = optim.Adam([generated], lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print("Starting neural style transfer training...")
    for step in range(2000):
        # Get features for generated, original, and style images
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = content_loss = 0
        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape
            content_loss += torch.mean((gen_feature - orig_feature)**2)

            # Compute Gram Matrix for style features
            G_gen = gen_feature.view(channel, height * width).matmul(
                gen_feature.view(channel, height * width).transpose(0, 1)
            )
            G_style = style_feature.view(channel, height * width).matmul(
                style_feature.view(channel, height * width).transpose(0, 1)
            )
            style_loss += torch.mean((G_gen - G_style)**2)

        total_loss = 0.001 * style_loss + 1 * content_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item():.4f}")
            save_image(generated, generated_img_path)

    print("Neural style transfer finished. Generated image saved.")

# --- Main Execution Flow --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer with PyTorch.")
    parser.add_argument("--content", type=str, default=CONTENT_PATH,
                        help="Path to the content image.")
    parser.add_argument("--style", type=str, default=STYLE_PATH,
                        help="Path to the style image.")
    parser.add_argument("--output", type=str, default=GENERATED_PATH,
                        help="Path to save the generated image.")
    args = parser.parse_args()

    # Create dummy content and style images for demonstration if they don't exist
    if not os.path.exists(args.content):
        dummy_content = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color = "red")
        dummy_content.save(args.content)
        print(f"Created dummy content image: {args.content}")
    if not os.path.exists(args.style):
        dummy_style = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color = "blue")
        dummy_style.save(args.style)
        print(f"Created dummy style image: {args.style}")

    train_neural_style_transfer(args.content, args.style, args.output)
    print("Neural style transfer example finished.")
