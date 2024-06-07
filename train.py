import torch
import torch.nn as nn
# from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Dataset
from going_modular import engine
from torch.hub import load_state_dict_from_url
from going_modular import engine
from models.HSNet import HSNet_model

# Call device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

class HierarchicalClassificationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_paths = self.get_img_paths()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_img_paths(self):
        img_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        level1_label = int(os.path.basename(os.path.dirname(img_path)))
        level2_label = int(os.path.basename(os.path.dirname(os.path.dirname(img_path))))

        return image, level2_label, level1_label


train_load = HierarchicalClassificationDataset('./dataset/dataset_v2/train')
val_load = HierarchicalClassificationDataset('./dataset/dataset_v2/val')

train_dataloader = DataLoader(train_load, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_load, batch_size=16, shuffle=False)
model = HSNet_model(pretrained=True).to(device)

# Defined Loss function and Optimizer for model

# lambda_lr = lambda epoch: 0.95 ** epoch
loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00001, amsgrad=True)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

result = engine.train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=val_dataloader,
                      optimizer=optimizer,
                      loss_fn1=loss_fn1,
                      loss_fn2=loss_fn2,
                      epochs=30,
                      device=device)
path_model = './weights/HSNet_make_Adam.pt'
torch.save(model.state_dict(), path_model)
model.load_state_dict(torch.load(path_model))
model.eval()


# Print plot
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(result["train_loss"], label="Train Loss")
plt.plot(result["test_loss"], label='Validation Loss')
plt.title("Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,3,2)
plt.plot(result["train_acc1"], label="Train Accuracy 1")
plt.plot(result["test_acc1"], label='Validation Accuracy 1')
plt.title("Accuracy 1")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,3,3)
plt.plot(result["train_acc2"], label="Train Accuracy 2")
plt.plot(result["test_acc2"], label='Validation Accuracy 2')
plt.title("Accuracy 2")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('./image/res50_preTrain_Coarse.png')
plt.show()
