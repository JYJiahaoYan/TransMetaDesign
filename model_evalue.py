import torch
import torch.nn as nn
import torchvision
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import interpolate
from model_evalue_plot import (
    double_ellipse_draw,
    double_rectangle_draw,
    circle_draw,
    rec_draw,
    cross_draw,
    lack_rec_draw,
    ring_draw
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ResNet50 Model Definition
class ResNet50Model(nn.Module):
    def __init__(self, output_size):
        super(ResNet50Model, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)


# Preprocess the Evaluation Data
def preprocess_evaluate(pred_data_path):
    with open(pred_data_path, "rb") as file:
        pred_data = pickle.load(file)

    label_data = pred_data["input_data"]
    pred_data = pred_data["pred"]

    types, args = [], []
    for pred in tqdm(pred_data):
        try:
            split_data = pred.split()
            pred_type, pred_args = parse_prediction(split_data)
            types.append(pred_type)
            args.append(pred_args)
        except Exception:
            types.append("Error")
            args.append(["Error"])

    return types, args, label_data


def parse_prediction(split_data):
    """Helper function to parse the prediction based on its type."""
    pred_type = split_data[0]
    if pred_type == "cross":
        W1 = float(split_data[7])
        L1 = float(split_data[11])
        W2 = float(split_data[15])
        L2 = float(split_data[19])
        offset = float(split_data[23])
        phi = float(split_data[27])
        Px = float(split_data[31])
        Py = float(split_data[35])
        return pred_type, [W1, L1, W2, L2, offset, Px, Py, phi]
    elif pred_type == "rec":
        W = float(split_data[7])
        L = float(split_data[11])
        phi = float(split_data[15])
        Px = float(split_data[19])
        Py = float(split_data[23])
        return pred_type, [L, W, Px, Py, phi]
    elif pred_type == "ellipse":
        a = float(split_data[7])
        b = float(split_data[11])
        phi = float(split_data[15])
        Px = float(split_data[19])
        Py = float(split_data[23])
        return pred_type, [a, b, Px, Py, phi]
    elif pred_type == "double_rec":
        W1 = float(split_data[7])
        L1 = float(split_data[11])
        W2 = float(split_data[15])
        L2 = float(split_data[19])
        phi = float(split_data[23])
        Px = float(split_data[27])
        Py = float(split_data[31])
        return pred_type, [W1, L1, W2, L2, Px, Py, phi]
    elif pred_type == "double_ellipse":
        a = float(split_data[7])
        b = float(split_data[11])
        theta = float(split_data[15])
        phi = float(split_data[19])
        Px = float(split_data[23])
        Py = float(split_data[27])
        return pred_type, [a, b, theta, Px, Py, phi]
    elif pred_type == "lack_rec":
        W = float(split_data[7])
        L = float(split_data[11])
        alpha = float(split_data[15])
        beta = float(split_data[19])
        gamma = float(split_data[23])
        phi = float(split_data[27])
        Px = float(split_data[31])
        Py = float(split_data[35])
        return pred_type, [L, W, alpha, beta, gamma, Px, Py, phi]
    elif pred_type == "ring":
        R = float(split_data[7])
        r = float(split_data[11])
        theta = float(split_data[15])
        phi = float(split_data[19])
        Px = float(split_data[23])
        Py = float(split_data[27])
        return pred_type, [R, r, theta, phi, Px, Py]
    else:
        raise ValueError(f"Unsupported type: {pred_type}")


# Spectrum Interpolation
def spectrum_interpolation(x, y, x_new):
    f = interpolate.interp1d(x, y, kind='linear', axis=1)
    return f(x_new)


# Model Prediction Function
def model_predict(img_batch, model_path):
    model = ResNet50Model(output_size=100).to(DEVICE)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    img_batch = img_batch.to(DEVICE)
    wave_pred_all = []

    with torch.no_grad():
        for i in tqdm(range(0, len(img_batch), 500)):
            wave_pred = model(img_batch[i:i + 500]).cpu().numpy()
            wave_pred_all.append(wave_pred)

    return np.concatenate(wave_pred_all, axis=0)


# Drawing Function Based on Shape Type
def draw_shape_by_type(shape_type, args):
    """Helper function to draw different shapes based on the type."""
    if shape_type == "cross":
        return cross_draw(args)
    elif shape_type == "rec":
        return rec_draw(args)
    elif shape_type == "ellipse":
        return circle_draw(args)
    elif shape_type == "double_rec":
        return double_rectangle_draw(args)
    elif shape_type == "double_ellipse":
        return double_ellipse_draw(args)
    elif shape_type == "lack_rec":
        return lack_rec_draw(args)
    elif shape_type == "ring":
        return ring_draw(args)
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")


def main(pred_data_path, model_path):
    types, args, labels = preprocess_evaluate(pred_data_path)

    pred_tensors, valid_labels = [], []
    for i, (type_, arg) in enumerate(zip(types, args)):
        if type_ == "Error" or arg[0] == "Error":
            continue
        valid_labels.append(labels[i])
        img_tensor = draw_shape_by_type(type_, arg)
        pred_tensors.append(img_tensor)

    img_batch = torch.stack(pred_tensors)
    wave_pred_all = model_predict(img_batch, model_path)

    light_source = np.linspace(400, 800, 100)
    light_source_new = np.linspace(400, 800, 500)
    preds = spectrum_interpolation(light_source, wave_pred_all, light_source_new)

    return preds


# Example usage:
if __name__ == "__main__":
    preds = main(
        r"D:\\codes\\optical_model_final\\eval_data_VIT3_epoch_12.pkl",
        r'D:\\codes\\data_autmention_model_final\\runs\\xxxx_best.pth'
    )
