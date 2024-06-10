from google.colab import files
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gdown
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from collections import Counter
import random
import zipfile
from imblearn.over_sampling import RandomOverSampler
import math
import shutil
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)
generator = torch.Generator(device=DEVICE)

"""# 1.Data upload"""


uploaded = files.upload()

def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        return zip_ref.namelist()

train_name = extract_zip('rokas_train.zip', 'train_data')
valid_name = extract_zip('rokas_validate.zip', 'validate_data')

"""# 2. Data representing """

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.images_list = []

        for files in os.listdir(dir):
            if files.endswith('.jpg'):
                parts = files.split('_')
                fing_nr = int(parts[0])
                self.images_list.append((os.path.join(dir, files),fing_nr))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path, fing_nr = self.images_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, fing_nr


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((-90, -90))
    ]),
    transforms.RandomAffine(    
        degrees=0,
        translate=(0.15, 0.15)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset("train_data/", transform=train_transform)
valid_dataset = CustomDataset("validate_data/", transform=valid_transform)


def denormalize(image, mean, std):
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image



def show_images_from_each_class(dataset, num_images=3, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    images_to_show = {i: [] for i in range(1, 6)}

    for image, label in dataset:
        label_value = label.item() if isinstance(label, torch.Tensor) else label
        if len(images_to_show[label_value]) < num_images:
            images_to_show[label_value].append(image)

    print("Images from each class:")
    for class_label, images in images_to_show.items():
        print(f"Class {class_label}:")
        plt.figure(figsize=(15, 5))
        for i, img in enumerate(images, 1):
            img = denormalize(img, mean, std)  
            img = img.permute(1, 2, 0).numpy()  
            img = np.clip(img, 0, 1)  
            plt.subplot(1, num_images, i)
            plt.imshow(img)
            plt.axis('off')
        plt.show()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


show_images_from_each_class(train_dataset, 3, mean, std)

def show_data_statistics(dataset):
    class_counts = {label: 0 for label in range(1, 6)}
    for _, label in dataset:
        label_value = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[label_value] += 1

    print("Data Statistics:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} images")


show_data_statistics(train_dataset)
show_data_statistics(valid_dataset)

"""# CNN Model"""

class FingersCount(nn.Module):
    def __init__(self, nr_classes=5):
        super(FingersCount, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(1024 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, nr_classes)

    def forward(self, x):

        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(nn.ReLU()(self.bn4(self.conv4(x))))
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 1024 * 4 * 4)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = FingersCount().to(DEVICE)

"""# Trainig"""


def eval_model(model, dataset, title="Validation Results"):
    validation_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    model.eval()
    model.to(DEVICE)
    all_estimations = []
    correct_estimations = []
    incorrect_estimations = []
    notzero = 1e-5

    with torch.no_grad():   
        for img, actual_count in dataset:
            img = img.to(DEVICE)
            predicted_count = torch.argmax(model(img.unsqueeze(0)), dim=1).item() + 1   
            all_estimations.append(predicted_count)
            if predicted_count == actual_count:
                correct_estimations.append((img, actual_count, predicted_count))
            else:
                incorrect_estimations.append((img, actual_count, predicted_count))

    total_correct = len(correct_estimations)
    total_samples = len(dataset)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Total Samples: {total_samples}, Correct Estimations: {total_correct}')

    def plot_confusion_matrix(model, dataset_loader):

      model.eval()
      all_labels = []
      all_predictions = []

      with torch.no_grad():
          for images, labels in dataset_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

      cm = confusion_matrix(all_labels, all_predictions)
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
      plt.xlabel("Predicted labels")
      plt.ylabel("True labels")
      plt.title("Confusion Matrix")
      plt.show()

    plot_confusion_matrix(model, validation_loader)

    if correct_estimations:
        display_images(correct_estimations, title, 'green', notzero)
    if incorrect_estimations:
        display_images(incorrect_estimations, title + " - Incorrect Estimations", 'red', notzero)

def display_images(data, title, color, notzero):
    num_images = len(data)
    num_columns = 5
    num_rows = math.ceil(num_images / num_columns)
    plt.figure(figsize=(15, 3 * num_rows))
    plt.suptitle(title, fontsize=16, verticalalignment='top')

    for idx, (img, actual_count, predicted_count) in enumerate(data):
        img_data = img.permute(1, 2, 0).cpu().numpy()
        img_data = img_data - np.min(img_data)
        img_data = img_data / (np.max(img_data) + notzero)
        ax = plt.subplot(num_rows, num_columns, idx + 1)
        ax.imshow(img_data)
        ax.set_title(f'Actual: {actual_count}, Predicted: {predicted_count}', fontsize=12, color=color)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def Finger_count_train(training_data, validation_data, num_epochs=100, batch_size=8, patience=20):

    model = FingersCount()
    model.to(DEVICE)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=DEVICE))
    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    model = FingersCount().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=3e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    epoch_list = []
    loss_list = []
    train_accuracy_list = []
    validation_accuracy_list = []
    best_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):   
        model.train()
        running_loss = 0.0
        correct_training = 0
        total_training = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE) - 1

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            running_loss += loss.item()
            _, predicted_training = torch.max(outputs.data, 1)
            total_training += labels.size(0)
            correct_training += (predicted_training == labels).sum().item()

        train_accuracy = 100 * correct_training / total_training
        print(f"Epoch: {epoch + 1}/{num_epochs}, | Loss: {running_loss / len(train_loader):.4f}, | Train Accuracy: {train_accuracy:.2f}% ")

        model.eval()
        correct_validation = 0
        total_validation = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE) - 1
                outputs = model(images)
                _, predicted_validation = torch.max(outputs.data, 1)
                total_validation += labels.size(0)
                correct_validation += (predicted_validation == labels).sum().item()

        validation_accuracy = 100 * correct_validation / total_validation
        lr_scheduler.step()
        epoch_list.append(epoch + 1)
        loss_list.append(running_loss / len(train_loader))
        train_accuracy_list.append(train_accuracy)
        validation_accuracy_list.append(validation_accuracy)

        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            if os.path.exists('finger_count.pth'):
                os.remove('finger_count.pth')
            torch.save(model.state_dict(), 'finger_count.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience and train_accuracy > best_acc:
            print('Game over.')
            break

    eval_model(model, valid_dataset, title="Final Validation")

    def plot_confusion_matrix(model, dataset_loader):

      model.eval()
      all_labels = []
      all_predictions = []

      with torch.no_grad():
          for images, labels in dataset_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

      cm = confusion_matrix(all_labels, all_predictions)
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
      plt.xlabel("Predicted labels")
      plt.ylabel("True labels")
      plt.title("Confusion Matrix")
      plt.show()

    plot_confusion_matrix(model, validation_loader)

"""# Accuracy """

TRAIN = False;  #for the trainig use true, if you want to upload existing training data, then false

if TRAIN:
    Finger_count_train(train_dataset, valid_dataset)
else:
    if not os.path.exists('finger_count.pth'):
        !wget -O finger_count.pth #path to your 

    if os.path.exists('finger_count.pth'):
        model.load_state_dict(torch.load('finger_count.pth', map_location=DEVICE))
    else:
        print("Nope, not there.")

    eval_model(model, valid_dataset)