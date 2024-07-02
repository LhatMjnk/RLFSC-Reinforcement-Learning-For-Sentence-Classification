import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, output_dim, name='ActorCriticNet', chkpt_dir='./Model/ActorCriticNet'):
        super(ActorCriticNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name)

        self.lstm = nn.LSTM(input_dim, 128, 2, batch_first=True, bidirectional=True, dropout=0.25)

        self.fc1 = nn.Linear(128*2, 256)
        self.fc2 = nn.Linear(256, 128)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.value = nn.Linear(128, 1)
        self.policy = nn.Linear(128, self.output_dim)

        self.layer_norm = nn.LayerNorm(128)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.lstm(state)

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.layer_norm(x)

        value = self.value(x)
        policy = self.policy(x)
        policy = F.softmax(policy, dim=-1)

        return value, policy
    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, name='Classifier', chkpt_dir='./Model/Classifier'):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name)

        self.lstm = nn.LSTM(input_dim, 128, 2, batch_first=True, bidirectional=True, dropout=0.25)

        self.fc1 = nn.Linear(128 * 2 * 50, 128)
        # self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(state)
        # x = state
        x = x.reshape(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = nn.Dropout(0.25)(x)
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, name='ActorNet', chkpt_dir='./Model/ActorNet'):
        super(ActorNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name)

        self.lstm = nn.LSTM(input_dim, 128, 3, batch_first=True, bidirectional=True, dropout=0.25)

        self.fc1 = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, 128)

        nn.init.xavier_uniform_(self.fc1.weight)

        self.policy = nn.Linear(128, self.output_dim)

        self.layer_norm = nn.LayerNorm(128)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(state)

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.layer_norm(x)

        policy = self.policy(x)
        policy = F.softmax(policy, dim=-1)

        return policy

class CriticClassifierNet(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, name='CriticClassifierNet', chkpt_dir='./Model/CriticClassifierNet'):
        super(CriticClassifierNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name)

        self.lstm = nn.LSTM(input_dim, 128, 3, batch_first=True, bidirectional=True, dropout=0.25)

        self.value_fc1 = nn.Linear(128*2, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)

        self.classfier_fc = nn.Linear(128*2*seq_len, 64)
        self.classifier = nn.Linear(64, output_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.lstm(state)

        value = F.gelu(self.value_fc1(x))
        value = F.gelu(self.value_fc2(value))
        value = self.value(value)

        x = x.reshape(x.size(0), -1)
        # Drop out
        x = nn.Dropout(0.5)(x)
        classifier = F.gelu(self.classfier_fc(x))
        classifier = self.classifier(classifier)

        return value, classifier