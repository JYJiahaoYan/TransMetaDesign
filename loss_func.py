from typing import Set

import editdistance
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torchmetrics import Metric
from torch import nn
from scipy.interpolate import interp1d
from torch.nn import init
from scaler import para_scaler
import numpy as np

import re


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")



# from data_generate.data_evalue import evalue
class CharacterErrorRate(Metric):
    full_state_update = False

    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.error: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]
            distance = editdistance.distance(pred, target)
            if max(len(pred), len(target)) > 0:
                self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self) -> Tensor:
        return self.error / self.total

class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            # nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        return self.layers(x)


class Critic_Loss(nn.Module):
    def __init__(self, ignore_index, tokenlizer):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.MSE = nn.L1Loss()
        self.tokenizer = tokenlizer

        self.ellipse_scaler, self.ellipse_model = self.get_config("ellipse")
        self.double_ellipse_scaler, self.double_ellipse_model = self.get_config("double_ellipse")
        self.double_rec_scaler, self.double_rec_model = self.get_config("double_rec")
        self.rec_scaler, self.rec_model = self.get_config("rec")
        self.ring_scaler, self.ring_model = self.get_config("ring")
        self.lack_rec_scaler, self.lack_rec_model = self.get_config("lack_rec")
        self.cross_scaler, self.cross_model = self.get_config("cross")

        self.pattern = r'(?<![a-zA-Z])-?\d+\.?\d*'  # 匹配浮点数的正则表达式

    def model_predic(self, logits, wave):
        para = [self.tokenizer.decode(i.tolist()) for i in torch.max(logits, dim=1)[1]]
        loss = 0
        punish_loss = 0
        wave = self.spec_interpolator(np.array(wave.cpu()))
        wave = torch.tensor(wave, dtype=torch.float32).to(DEVICE)
        wave = wave.unsqueeze(1)
        for idx,i in enumerate(para):
            try:
                shape = i.split()[0]
                split_data = re.findall(self.pattern,i)
                split_data = [float(i) for i in split_data]
                # [W, L, W1,L1, W2, L2, offset, alpha, beta, gamma, R, r, a, b, theta, phi, Px, Py]
                if shape == "double_rec":
                    split_data = torch.tensor(self.double_rec_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.double_rec_model.eval()
                    pred = self.double_rec_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "cross":
                    split_data = torch.tensor(self.cross_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.cross_model.eval()
                    pred = self.cross_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "lack_rec":
                    split_data = torch.tensor(self.lack_rec_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.lack_rec_model.eval()
                    pred = self.lack_rec_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "rec":
                    split_data = torch.tensor(self.rec_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.rec_model.eval()
                    pred = self.rec_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "ring":
                    split_data = torch.tensor(self.ring_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.ring_model.eval()
                    pred = self.ring_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "ellipse":
                    split_data = torch.tensor(self.ellipse_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.ellipse_model.eval()
                    pred = self.ellipse_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                elif shape == "double_ellipse":
                    split_data = torch.tensor(self.double_ellipse_scaler.transform(np.array(split_data).reshape(1, -1)),dtype=torch.float32).to(DEVICE)
                    self.double_ellipse_model.eval()
                    pred = self.double_ellipse_model(split_data, None)
                    loss = loss + self.MSE(pred, wave[idx]) * 100
                else:
                    raise ValueError("模型预测参数错误")
                return (loss / len(para))
            except:
                return torch.tensor(50.0)

    def get_config(self,shape):
        if shape == 'ellipse':
            min_val = np.array([25, 25, 0, 50, 50])
            max_val = np.array([400, 400, 180, 600, 600])
            input_dim = 5
        elif shape == "double_ellipse":
            min_val = np.array([240, 60, 0, 0, 300, 300])
            max_val = np.array([450, 200, 45, 360, 800, 800])
            input_dim = 6
        elif shape == "double_rec":
            min_val = np.array([50, 100, 50, 100, 0, 180, 180])
            max_val = np.array([300, 750, 300, 750, 180, 900, 900])
            input_dim = 7
        elif shape == "rec":
            min_val = np.array([25, 25, 0, 50, 50])
            max_val = np.array([400, 400, 180, 600, 600])
            input_dim = 5
        elif shape == "ring":
            min_val = np.array([50, 0, 0, 0, 100, 100])
            max_val = np.array([450, 450, 360, 360, 900, 900])
            input_dim = 6
        elif shape == "lack_rec":
            min_val = np.array([50, 50, 0, 0, 0, 0, 100, 100])
            max_val = np.array([700, 700, 1, 1, 1, 360, 900, 900])
            input_dim = 8
        elif shape == "cross":
            min_val = np.array([20, 150, 20, 150, -300, 0, 200, 200])
            max_val = np.array([200, 600, 200, 600, 300, 180, 800, 800])
            input_dim = 8

        scaler = MinMaxScaler()
        scaler.min_ = min_val
        scaler.scale_ = 1 / (max_val - min_val)

        forward_model = MLP(input_dim, 100).to(DEVICE)
        model_path = f'data/critic_model/forward_01_{shape}.pth'
        forward_model.load_state_dict(torch.load(model_path)['model_state_dict'])
        forward_model.eval()
        return scaler, forward_model


    def spec_interpolator(self,wave):
        interpolator = interp1d(np.linspace(400, 800, 500), wave, kind='cubic')
        new_wave = interpolator(np.linspace(400, 800, 100))
        return new_wave
    
    def forward(self, logits: Tensor, targets: Tensor, wave) -> Tensor:
        inverse_loss = 0
        targets = targets[:, 1:]
        mse_loss = self.model_predic(logits, wave)
        ce_loss = self.CE(logits, targets)
        mul = targets > 1000
        pred = torch.max(logits, dim=1)[1]
        # mse_loss = self.MSE((pred*mul).float(),(targets*mul).float())
        type_loss = self.MSE(pred[:, 0].float(), targets[:, 0].float())
        # mse_loss = self.MSE(targets.to(torch.float32),pred.to(torch.float32))
        inverse_loss = (ce_loss, mse_loss, type_loss)

        return inverse_loss



class Token_Loss(nn.Module):
    def __init__(self, ignore_index, tokenlizer):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.MSE = nn.L1Loss()
        self.tokenizer = tokenlizer


    def forward(self, logits: Tensor, targets: Tensor, wave) -> Tensor:
        [self.tokenizer.decode(i.tolist()) for i in torch.max(logits, dim=1)[1]]
        inverse_loss = 0
        targets = targets[:, 1:]
        ce_loss = self.CE(logits, targets)
        mul = targets > 1000
        pred = torch.max(logits, dim=1)[1]
        mse_loss = self.MSE((pred*mul).float(),(targets*mul).float())
        type_loss = self.MSE(pred[:, 0].float(), targets[:, 0].float())
        inverse_loss = (ce_loss, mse_loss, type_loss)

        return inverse_loss

class CE_Loss(nn.Module):
    def __init__(self, ignore_index, tokenlizer):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.MSE = nn.L1Loss()
        self.tokenizer = tokenlizer


    def forward(self, logits: Tensor, targets: Tensor, wave) -> Tensor:
        inverse_loss = 0
        targets = targets[:, 1:]
        ce_loss = self.CE(logits, targets)
        pred = torch.max(logits, dim=1)[1]
        mse_loss = torch.tensor(0.)
        type_loss = self.MSE(pred[:, 0].float(), targets[:, 0].float())
        inverse_loss = (ce_loss, mse_loss, type_loss)

        return inverse_loss

class MSE_Loss(nn.Module):
    def __init__(self, ignore_index, tokenlizer):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.MSE = nn.L1Loss()
        self.tokenizer = tokenlizer


    def forward(self, logits: Tensor, targets: Tensor, wave) -> Tensor:
        inverse_loss = 0
        targets = targets[:, 1:]
        ce_loss = self.CE(logits, targets) * torch.tensor(0.)
        mul = targets > 1000
        pred = torch.max(logits, dim=1)[1]
        mse_loss = self.MSE((pred*mul).float(),(targets*mul).float())
        type_loss = self.MSE(pred[:, 0].float(), targets[:, 0].float())
        inverse_loss = (ce_loss, mse_loss, type_loss)

        return inverse_loss
