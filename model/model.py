import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# functions of initializing layers
def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_rnn(rnn):
    """init_rnn
    Initialized RNN weights, independent of type GRU/LSTM/RNN
    :param rnn: the rnn model 
    """
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)

def reset_parameters(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            init_layer(module)
        elif isinstance(module, nn.Linear):
            init_layer(module)
        elif isinstance(module, nn.BatchNorm2d):
            init_bn(module)
        elif isinstance(module, nn.LSTM):
            init_rnn(module)

#############################################################################
#################below is the implementation of a simple cnn network#########

class simple_cnn(BaseModel):
    def __init__(self, num_classes = 2, in_channel=1):
        super(simple_cnn, self).__init__()
        self.in_channel = in_channel 
        self.conv1 = nn.Conv2d(in_channel,16,3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(64,32)
        self.output = nn.Linear(32,num_classes)
        self.activate = nn.Softmax()
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.output)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)


    def forward(self,x):
        B, mel_bins, num_frames = x.size()
        x = x.view(B, self.in_channel, -1, num_frames)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        x = self.bn3(x)
        x = self.pool(x).reshape(x.size(0),-1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.output(x)
        out = self.activate(out)
        return out

###########################################################################
################ a deeper CNN model, VGG like structure ################### 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = (3,3),
                               stride = (1,1),
                               padding = (1,1),
                               bias =False)

        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = (3,3),
                               stride = (1,1),
                               padding = (1,1),
                               bias = False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation = 'relu'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if activation == 'relu':
            x = F.relu_(self.bn2(self.conv2(x)))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class VGG_11(BaseModel):
    def __init__(self, num_classes, in_channel):
        super(VGG_11, self).__init__()
        self.in_channel = in_channel
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ConvBlock(in_channels=in_channel, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc_final = nn.Linear(512, num_classes)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_final)

    def forward(self, input):
        # (batch_size, 3, mel_bins, time_stamps)
        B, mel_bins, num_frames = input.size()
        x = input.view(B, self.in_channel, -1, num_frames)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1,2)

        # (samples_num, channel, mel_bins, time_stamps)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training = self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])
        output = F.log_softmax(self.fc_final(output), dim=-1)
        return output


class VGG_13(BaseModel):
    def __init__(self, num_classes, in_channel):
        super(VGG_13, self).__init__()
        self.in_channel = in_channel
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ConvBlock(in_channels=in_channel, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512)
        self.fc_1 = nn.Linear(512 * 4* 10, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_final = nn.Linear(4096, num_classes)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_final)
        init_layer(self.fc_1)
        init_layer(self.fc_2)

    def forward(self, input):
        # (batch_size, 3, mel_bins, time_stamps)
        B, mel_bins, num_frames = input.size()
        x = input.view(B, self.in_channel, -1, num_frames)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1,2)

        # (samples_num, channel, time_stemps, mel_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training = self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv5(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        #output = F.max_pool2d(x, kernel_size=x.shape[2:])
        #output = output.view(output.shape[0:2])
        x = x.view(x.shape[0], -1) 
        x = F.relu_(self.fc_1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc_2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        output = F.log_softmax(self.fc_final(x),dim=-1)
        #output = F.log_softmax(self.fc_final(output), dim=-1)
        return output

##############################################################################
################ below is the implementation of rnn model ####################
class BiLSTM(BaseModel):
    def __init__(self,
                 input_dim=200,
                 hidden_dim=256, 
                 num_layers =2,
                 dropout=0.2,
                 num_classes=2,
                 pooling='first',
                 model='lstm',
                 BN=False):
        super(BiLSTM,self).__init__()
        if model == 'lstm':
            self.LSTM = nn.LSTM(input_size=input_dim,
                                hidden_size =hidden_dim,
                                num_layers =num_layers,
                                batch_first =True, 
                                dropout=dropout,
                                bidirectional=True)
        elif model == 'gru':
            self.LSTM = nn.GRU(input_size=input_dim,
                               hidden_size =hidden_dim,
                               num_layers =num_layers,
                               batch_first =True, 
                               dropout=dropout,
                               bidirectional=True)
        init_rnn(self.LSTM)
        self.BN = BN
        if self.BN:
            self.BatchNorm=nn.BatchNorm1d(hidden_dim*2)
        
        self.layer_out = nn.Linear(hidden_dim*2,num_classes,bias=False)
        self.pooling=pooling

    def forward(self,x):
        x = x.transpose(2,1) #[Batchsize x Time_frames x Mel_bin]
        x,_ = self.LSTM(x) 
        if self.BN:
            x = self.BatchNorm(x.transpose(1,2))
            x = x.transpose(1,2)
        dim =1
        x = self.layer_out(x)  #200,2
        if self.pooling == 'avg':
            x = x.mean(dim)   #average pooling
        elif self.pooling == 'first':
            x = x.select(dim,0)  #first time step
        elif self.pooling == 'last':
            x = x.select(dim,-1)  #first time step
        elif self.pooling == 'max':
            x = x.max(dim)[0]
        elif  self.pooling == 'linear':
            (x**2).sum(dim) / x.sum(dim)
        elif self.pooling == 'exp':
            (x.exp() * x).sum(dim) / x.exp().sum(dim)
        
        # print(out.shape)
        # out = self.LogSoftmax(out)#the input given is expected to contain log-probabilities. Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        return x#,A        


#############################################################################
#################### below is the implementation of crnn network ############

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(CNN_Encoder, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=256)
    def forward(self, input):
        x = input
        x = self.conv1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training = self.training)
        x = self.conv4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        
        return x

class crnn(BaseModel):
    def __init__(self, in_channel, 
                 hidden_dim=128,
                 num_layers = 2,
                 dropout = 0.2,
                 pooling = 'first',
                 model='lstm',
                 BN = True,
                 num_classes=2):
        super(crnn,self).__init__()
        
        self.in_channel = in_channel
        self.bn0 = nn.BatchNorm2d(128)
 
        self.cnn = CNN_Encoder(in_channels=in_channel)
        self.bilstm = BiLSTM(input_dim=256,
                             hidden_dim=hidden_dim, 
                             num_layers =num_layers,
                             dropout =dropout,
                             num_classes=num_classes,
                             pooling=pooling,
                             model=model,
                             BN=BN)
        
    
    def forward(self,input):
        # (batch_size, 3, mel_bins, time_stamps)
        B, mel_bins, num_frames = input.size()
        x = input.view(B, self.in_channel, -1, num_frames)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1,2)
        
        x = self.cnn(x)
        x = F.max_pool2d(x,kernel_size=(x.size(-2), 1)) # pool mel dimension
        x = x.squeeze(-2)  # [B x C x Time_frames]
        output = self.bilstm(x)
        return output


