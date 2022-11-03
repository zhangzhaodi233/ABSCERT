from torch import nn

class FC3_Relu(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat


class FC5_Relu(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, int(linear_size / 4)),
            nn.ReLU(),
            nn.Linear(int(linear_size/4), 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat


class FC3_Sigmoid(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat



class FC5_Sigmoid(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, int(linear_size / 4)),
            nn.Sigmoid(),
            nn.Linear(int(linear_size/4), 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat


class FC3_Tanh(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat



class FC5_Tanh(nn.Module):
    def __init__(self, in_ch, in_dim, linear_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * in_dim * in_dim, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, int(linear_size / 4)),
            nn.Tanh(),
            nn.Linear(int(linear_size/4), 10)
        )
    
    def forward(self, x, y=None):
        y_hat = self.fc(x)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat


