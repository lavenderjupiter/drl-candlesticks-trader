import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

def get_conv_out(conv_block, shape):
    out = conv_block(torch.zeros(1, *shape))
    return int(np.prod(out.size()))

#------------ResNet with CBAM------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//reduction, in_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat_out)
        return self.sigmoid(out)

class DQN_HOOK(nn.Module):
    def __init__(self, dqn_model):
        super(DQN_HOOK, self).__init__()
        self.backbone = dqn_model.cnn.backbone
        self.ca = dqn_model.cnn.ca
        self.sa = dqn_model.cnn.sa
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = dqn_model.mlp
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        out = self.backbone(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return out

    def forward(self, x, account):
        out = self.backbone(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        h = out.register_hook(self.activations_hook)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        account = torch.flatten(account, 1)
        out = torch.cat((out, account), 1)
        out = self.mlp(out)
        return out


#------------Append CBAM to Backbone------------

class AddCBAM(nn.Module):
    def __init__(self,input_shape, backbone):
        super(AddCBAM, self).__init__()
        self.input_shape = input_shape
        self.backbone = backbone
        out_channel = self._get_conv_out_channel(self.input_shape)

        self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _get_conv_out_channel(self, shape):
        out = self.backbone(torch.zeros(1, *shape))
        return out.size()[1]

    def forward(self, x):
        out = self.backbone(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.avgpool(out)
        return out


#------------RL Nets------------

class DQNNet(nn.Module):
    def __init__(self, input_shape, n_actions, backbone, account_feats=3, raw_data='None'):
        super(DQNNet, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.cnn = backbone
        self.raw_data = raw_data
        if self.raw_data == 'None':
            con_out_size = get_conv_out(backbone, input_shape) + account_feats
        else:
            con_out_size = input_shape[0] + account_feats

        if self.raw_data == 's':
            self.mlp = nn.Sequential(
                nn.Linear(con_out_size, 10),
                nn.ReLU(), 
                nn.Linear(10, 5),
                nn.ReLU(), 
                nn.Linear(5, self.n_actions),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(con_out_size, 512),
                nn.ReLU(), 
                nn.Linear(512, 256),
                nn.ReLU(), 
                nn.Linear(256, self.n_actions),
            )

    def forward(self, x, current_account):   
        if self.raw_data == 'None':
            out = self.cnn(x)
            out = torch.flatten(out, 1) 
            current_account = torch.flatten(current_account, 1)
            out = torch.cat((out, current_account), 1)
        else:
            out = torch.cat((x, current_account), 1)
        out = self.mlp(out)
        return out

class DuelingNet(nn.Module):
    def __init__(self, input_shape, n_actions, backbone, account_feats=3, device='cpu', raw_data='None'):
        super(DuelingNet, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.cnn = backbone
        self.device = device
        self.raw_data = raw_data
        if self.raw_data == 'None':
            con_out_size = get_conv_out(backbone, input_shape) + account_feats
        else:
            con_out_size = input_shape[0] + account_feats

        if self.raw_data == 's':
            self.value = nn.Sequential(
                nn.Linear(con_out_size, 10),
                nn.ReLU(), 
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1),          
            )
            self.advantage = nn.Sequential(
                nn.Linear(con_out_size, 10),
                nn.ReLU(), 
                nn.Linear(10, 5),
                nn.ReLU(), 
                nn.Linear(5, self.n_actions),
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(con_out_size, 512),
                nn.ReLU(), 
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),          
            )
            self.advantage = nn.Sequential(
                nn.Linear(con_out_size, 512),
                nn.ReLU(), 
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_actions),    
            )

    def forward(self, x, current_account, legal_actions=None):  
        if self.raw_data == 'None':
            out = self.cnn(x)
            out = torch.flatten(out, 1) 
            current_account = torch.flatten(current_account, 1)
            out = torch.cat((out, current_account), 1)
        else:
            out = torch.cat((x, current_account), 1)
        value = self.value(out)
        advantage = self.advantage(out)

        if legal_actions is not None:
            action_mask = torch.zeros((len(x), self.n_actions))
            for i, a in enumerate(legal_actions):
                mask = torch.zeros((1, self.n_actions))
                mask[0, a] = 1
                action_mask[i] = mask            
            advantage[action_mask == 0] = -9999999
            adv_mean = torch.tensor([a[a!=-9999999].mean() for a in advantage]).unsqueeze(-1)
            out = value + advantage - adv_mean.to(self.device)
        else:        
            out = value + advantage - torch.mean(advantage, dim=1).unsqueeze(-1)
        return out


class ActorPPO(nn.Module):
    def __init__(self, input_shape, n_actions, backbone, account_feats=3, cont=False, std=0.0, raw_data='None'):
        super(ActorPPO, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.continuous = cont
        self.cnn = backbone
        self.raw_data = raw_data
        if self.raw_data == 'None':
            con_out_size = get_conv_out(backbone, input_shape) + account_feats
        else:
            con_out_size = input_shape[0] + account_feats

        if self.raw_data == 's':
            self.actor = nn.Sequential(
                nn.Linear(con_out_size, 10),
                nn.ReLU(), 
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, self.n_actions),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(con_out_size, 512),
                nn.ReLU(), 
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_actions),
            )  


    def forward(self, x, current_account, action_mask):
        if self.raw_data == 'None':
            out = self.cnn(x)
            out = torch.flatten(out, 1) 
            current_account = torch.flatten(current_account, 1)
            out = torch.cat([out, current_account], 1)
        else:
            out = torch.cat([x, current_account], 1)

        output = self.actor(out)
        output[action_mask == 0] = -float('inf')
        prob = F.softmax(output, dim=-1)#.squeeze(0)
        dist = Categorical(prob)
        return dist

class CriticPPO(nn.Module):
    def __init__(self, input_shape, backbone, account_feats=3, raw_data=False):
        super(CriticPPO, self).__init__()
        self.input_shape = input_shape
        self.cnn = backbone
        self.raw_data = raw_data
        if self.raw_data == 'None':
            con_out_size = get_conv_out(backbone, input_shape) + account_feats
        else:
            con_out_size = input_shape[0] + account_feats

        if self.raw_data == 's':
            self.critic = nn.Sequential(
                nn.Linear(con_out_size, 10),
                nn.ReLU(), 
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)           
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(con_out_size, 512),
                nn.ReLU(), 
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)           
            )

    def forward(self, x, current_account):
        if self.raw_data == 'None':
            out = self.cnn(x)
            out = torch.flatten(out, 1) 
            current_account = torch.flatten(current_account, 1)
            out = torch.cat([out, current_account], 1)
        else:
            out = torch.cat([x, current_account], 1)

        value = self.critic(out)
        return value
