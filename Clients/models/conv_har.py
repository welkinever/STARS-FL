import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param


class HARConv1D(nn.Module):
    """Modified Conv model for HAR with 1D convolution."""
    def __init__(self, input_channel=1, hidden_size=[64, 64, 64], num_classes=6, rate=1, track=False):
        super(HARConv1D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, hidden_size[0], kernel_size=5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(hidden_size[0], hidden_size[1], kernel_size=5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(hidden_size[1], hidden_size[2], kernel_size=5),
            nn.ReLU(),
        )
        
        flattened_size = 35136

        if 'layerwise' in cfg['algo'] and rate <= 1 and cfg['global_model_rate'] == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(int(flattened_size*rate*cfg['layerwise_ratio'][-1]), int(128*rate)),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(int(128*rate), num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(int(flattened_size*rate), int(128*rate)),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(int(128*rate), num_classes),
            )
    
    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        out = self.classifier(x)
        output['score'] = out
        output['loss'] = F.cross_entropy(out, input['label'], reduction='mean')
        return output


def conv_har(model_rate=1, track=False):
    input_channel = 1
    if model_rate <= 1 and 'layerwise' in cfg['algo']:
        hidden_size = [int(np.ceil(model_rate * x * y)) for x, y in zip([64,64,64], cfg['layerwise_ratio'])]
    else:
        hidden_size = [int(np.ceil(model_rate * x)) for x in [64,64,64]]
    num_classes = cfg['classes_size']
    model = HARConv1D(input_channel, hidden_size, num_classes, model_rate, track)
    model.apply(init_param)
    return model
