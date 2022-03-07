from .resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from resnet import resnet18, resnet34

class Res_Attention(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(Res_Attention, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = ResNet10()

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):
        if debug:
            print("input shape:", x.shape)
        x = x.squeeze(0)
        if debug:
            print("squeeze shape:", x.shape)

        H = self.feature_extractor_part1(x)
        if debug:
            print("feature_extractor_part1 shape:", H.shape)
        H = H.view(-1, 512 * 7 * 7)
        if debug:
            print("view shape:", H.shape)
        H = self.feature_extractor_part2(H)  # NxL
        if debug:
            print("feature_extractor_part2 shape:", H.shape)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #print("Y_prob is：", Y_prob)
        return  Y_prob, Y_hat ,neg_log_likelihood

class VideoModel1(nn.Module):
    def __init__(self, num_class=2):
        super(VideoModel1, self).__init__()
        self.backbone = torchvision.models.video.r3d_18(pretrained = True)
        self.classifier = nn.Sequential(
            nn.Linear(400, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            # nn.Linear(400,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x



class VideoModel2(nn.Module):
    def __init__(self,):
        super(VideoModel2,self).__init__()
        #self.norm = nn.LayerNorm(11)
        self.cnn = torchvision.models.resnet18(pretrained = True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, 64)
        self.l1 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers = 3,
            batch_first = True
        )
        
        self.out = nn.Sequential(
            nn.Linear(64*16,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        #x = self.norm(x)
        #x:BCTHW
        out_features = torch.ones((x.shape[0], x.shape[2], 64)).cuda()
        for i in range(out_features.shape[0]):
            out_features[i] = self.cnn(x[i].permute(1,0,2,3))
        r_out, h = self.l1(out_features)
        final_features = r_out.contiguous().view(r_out.shape[0], -1)
        out = self.out(final_features)
        
        return out

if __name__ == '__main__':
    model = Res_Attention()

