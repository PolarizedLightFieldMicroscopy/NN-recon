"""Script to test a trained model on the set of birefringence data"""

import torch
from torchsummary import summary
from tifffile import imread, imwrite
import os
from data import BirefringenceDataset
from model_bir import BirNetwork, BirNetwork1, BirNetworkDense

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using {device} device")

saved_model_dir = "/mnt/efs/shared_data/restorators/models_bir/"

DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
test_data = BirefringenceDataset(
    DATA_PATH, split="test", source_norm=True, target_norm=True
)
testloader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=2
)

model_relu = BirNetwork1().to(device)
print(summary(model_relu, (512, 16, 16)))
model_relu.eval()
weights_relu = torch.load(saved_model_dir + "sphere128.pt")
model_relu.load_state_dict(weights_relu)

# model_leaky = BirNetwork().to(device)
# model_leaky.eval()
# weights_leaky = torch.load(saved_model_dir + 'sphere_9_2_epoch67.pt')
# model_leaky.load_state_dict(weights_leaky)

# model_normed = BirNetwork().to(device)
# model_normed.eval()
# weights_normed = torch.load(saved_model_dir + 'sphere_9_3_norm/norm_final.pt')
# model_normed.load_state_dict(weights_normed)

# model_normed_ker5 = BirNetwork().to(device)
# model_normed_ker5.eval()
# weights_normed_ker5 = torch.load(saved_model_dir + 'sphere_9_3_norm_ker5/norm_final.pt')
# model_normed_ker5.load_state_dict(weights_normed_ker5)

data_pair = test_data[0]
source = data_pair[0]
source = source.unsqueeze(axis=0).to(device)
with torch.no_grad():
    target_pred_relu = model_relu(source).cpu().squeeze(axis=0).numpy()
    # target_pred_leaky = model_leaky(source).cpu()
    # target_pred_normed = model_normed(source).cpu()
    # target_pred_normed_ker5 = model_normed_ker5(source).cpu()

save_mode = False
if save_mode:
    save_dir = "inference/mymodel/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    imwrite(os.path.join(save_dir, "pred.tiff"), target_pred_relu)
    source = source.cpu().squeeze(axis=0).numpy()
    imwrite(os.path.join(save_dir, "source.tiff"), source)
