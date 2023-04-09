# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:16:52.669904Z","iopub.execute_input":"2023-04-09T14:16:52.670354Z","iopub.status.idle":"2023-04-09T14:16:52.678146Z","shell.execute_reply.started":"2023-04-09T14:16:52.670317Z","shell.execute_reply":"2023-04-09T14:16:52.677028Z"}}
import os
import itertools
import time
import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:16:52.997708Z","iopub.execute_input":"2023-04-09T14:16:52.998140Z","iopub.status.idle":"2023-04-09T14:16:53.007061Z","shell.execute_reply.started":"2023-04-09T14:16:52.998100Z","shell.execute_reply":"2023-04-09T14:16:53.005576Z"}}
class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '../input/mitbih-with-synthetic/attn.pth'
    lstm_state_path = '../input/mitbih-with-synthetic/lstm.pth'
    cnn_state_path = '../input/mitbih-with-synthetic/cnn.pth'
    
    attn_logs = '../input/mitbih-with-synthetic/attn.csv'
    lstm_logs = '../input/mitbih-with-synthetic/lstm.csv'
    cnn_logs = '../input/mitbih-with-synthetic/cnn.csv'
    
    train_csv_path = '../input/mitbih-with-synthetic/mitbih_with_syntetic_train.csv'
    test_csv_path = '../input/mitbih-with-synthetic/mitbih_with_syntetic_test.csv'

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
config = Config()
seed_everything(config.seed)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:16:53.289514Z","iopub.execute_input":"2023-04-09T14:16:53.290112Z","iopub.status.idle":"2023-04-09T14:16:57.817183Z","shell.execute_reply.started":"2023-04-09T14:16:53.290076Z","shell.execute_reply":"2023-04-09T14:16:57.815750Z"}}
df_ptbdb = pd.read_csv('/kaggle/input/heartbeat/ptbdb_abnormal.csv')
df_mitbih = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv')
df_ptbdb

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:16:57.819872Z","iopub.execute_input":"2023-04-09T14:16:57.820733Z","iopub.status.idle":"2023-04-09T14:16:57.826688Z","shell.execute_reply.started":"2023-04-09T14:16:57.820682Z","shell.execute_reply":"2023-04-09T14:16:57.825881Z"}}
len(df_ptbdb)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:16:57.827828Z","iopub.execute_input":"2023-04-09T14:16:57.828405Z","iopub.status.idle":"2023-04-09T14:17:02.771059Z","shell.execute_reply.started":"2023-04-09T14:16:57.828365Z","shell.execute_reply":"2023-04-09T14:17:02.770196Z"}}
df_mitbih_train = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header=None)
df_mitbih_test = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header=None)
df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)
df_mitbih.rename(columns={187: 'class'}, inplace=True)

id_to_label = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}
df_mitbih['label'] = df_mitbih.iloc[:, -1].map(id_to_label)
print(df_mitbih.info())

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:17:02.772277Z","iopub.execute_input":"2023-04-09T14:17:02.772715Z","iopub.status.idle":"2023-04-09T14:17:02.777848Z","shell.execute_reply.started":"2023-04-09T14:17:02.772682Z","shell.execute_reply":"2023-04-09T14:17:02.776912Z"}}
len(df_mitbih_test)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:17:02.781096Z","iopub.execute_input":"2023-04-09T14:17:02.781394Z","iopub.status.idle":"2023-04-09T14:17:26.594014Z","shell.execute_reply.started":"2023-04-09T14:17:02.781365Z","shell.execute_reply":"2023-04-09T14:17:26.593012Z"}}
df_mitbih.to_csv('data.csv', index=False)
config.csv_path = 'data.csv'

# %% [markdown]
# # Basic EDA

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:17:26.597303Z","iopub.execute_input":"2023-04-09T14:17:26.597637Z","iopub.status.idle":"2023-04-09T14:17:29.809964Z","shell.execute_reply.started":"2023-04-09T14:17:26.597604Z","shell.execute_reply":"2023-04-09T14:17:29.808825Z"}}
df_mitbih = pd.read_csv(config.csv_path)
df_mitbih['label'].value_counts()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T14:17:29.811742Z","iopub.execute_input":"2023-04-09T14:17:29.812165Z","iopub.status.idle":"2023-04-09T14:17:30.430963Z","shell.execute_reply.started":"2023-04-09T14:17:29.812029Z","shell.execute_reply":"2023-04-09T14:17:30.429761Z"}}
percentages = [count / df_mitbih.shape[0] * 100 for count in df_mitbih['label'].value_counts()]

fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(
    x=df_mitbih['label'],
    ax=ax,
    palette="bright",
    order=df_mitbih['label'].value_counts().index
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15);

for percentage, count, p in zip(
    percentages,
    df_mitbih['label'].value_counts(sort=True).values,
    ax.patches):
    
    percentage = f'{np.round(percentage, 2)}%'
    x = p.get_x() + p.get_width() / 2 - 0.4
    y = p.get_y() + p.get_height()
    ax.annotate(str(percentage)+" / "+str(count), (x, y), fontsize=12, fontweight='bold')
    
plt.savefig('data_dist.png', facecolor='w', edgecolor='w', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.savefig('data_dist.svg', facecolor='w', edgecolor='w', format='svg',
        transparent=False, bbox_inches='tight', pad_inches=0.1)

# %% [markdown]
# I used the GAN from the notebook you can find [here](https://www.kaggle.com/polomarco/1d-gan-for-ecg-synthesis) or a repository with the code [here](https://github.com/mandrakedrink/ECG-Synthesis-and-Classification) to generate new synthetic data for classes with little data, now the dataset looks like this:

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:17:30.432589Z","iopub.execute_input":"2023-04-09T14:17:30.433410Z","iopub.status.idle":"2023-04-09T14:17:34.085907Z","shell.execute_reply.started":"2023-04-09T14:17:30.433356Z","shell.execute_reply":"2023-04-09T14:17:34.084741Z"}}
config.csv_path = '../input/mitbih-with-synthetic/mitbih_with_syntetic.csv'
df_mitbih_new = pd.read_csv(config.csv_path)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T14:17:34.087775Z","iopub.execute_input":"2023-04-09T14:17:34.088111Z","iopub.status.idle":"2023-04-09T14:17:35.140666Z","shell.execute_reply.started":"2023-04-09T14:17:34.088079Z","shell.execute_reply":"2023-04-09T14:17:35.139661Z"}}
percentages1 = [count / df_mitbih.shape[0] * 100 for count in df_mitbih['label'].value_counts()]
percentages2 = [count / df_mitbih_new.shape[0] * 100 for count in df_mitbih_new['label'].value_counts()]

fig, axs = plt.subplots(1,2, figsize=(18, 4))

# origin
sns.countplot(
    x=df_mitbih['label'],
    ax=axs[0],
    palette="bright",
    order=df_mitbih['label'].value_counts().index
)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=15);
axs[0].set_title("Before", fontsize=15)

for percentage, count, p in zip(
    percentages1,
    df_mitbih['label'].value_counts(sort=True).values,
    axs[0].patches):
    
    percentage = f'{np.round(percentage, 2)}%'
    x = p.get_x() + p.get_width() / 2 - 0.4
    y = p.get_y() + p.get_height()
    axs[0].annotate(str(percentage)+" / "+str(count), (x, y), fontsize=10, fontweight='bold')

# with synthetic
sns.countplot(
    x=df_mitbih_new['label'],
    ax=axs[1],
    palette="bright",
    order=df_mitbih_new['label'].value_counts().index
)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=15);
axs[1].set_title("After", fontsize=15)

for percentage, count, p in zip(
    percentages2,
    df_mitbih_new['label'].value_counts(sort=True).values,
    axs[1].patches):
    
    percentage = f'{np.round(percentage, 2)}%'
    x = p.get_x() + p.get_width() / 2 - 0.4
    y = p.get_y() + p.get_height()
    axs[1].annotate(str(percentage)+" / "+str(count), (x, y), fontsize=10, fontweight='bold')

#plt.suptitle("Balanced Sampling between classes", fontsize=20, weight="bold", y=1.01)
plt.savefig('data_dist.png', facecolor='w', edgecolor='w', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.savefig('data_dist.svg', facecolor='w', edgecolor='w', format='svg',
        transparent=False, bbox_inches='tight', pad_inches=0.1)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T14:17:35.142027Z","iopub.execute_input":"2023-04-09T14:17:35.142323Z","iopub.status.idle":"2023-04-09T14:17:37.283196Z","shell.execute_reply.started":"2023-04-09T14:17:35.142293Z","shell.execute_reply":"2023-04-09T14:17:37.282109Z"}}
N = 5
samples = [df_mitbih.loc[df_mitbih['class'] == cls].sample(N) for cls in range(N)]
titles = [id_to_label[cls] for cls in range(5)]

with plt.style.context("seaborn-white"):
    fig, axs = plt.subplots(3, 2, figsize=(20, 7))
    for i in range(5):
        ax = axs.flat[i]
        ax.plot(samples[i].values[:,:-2].transpose())
        ax.set_title(titles[i])
        #plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")
    plt.savefig(f"signals_per_class.svg",
                    format="svg",bbox_inches='tight', pad_inches=0.2)
        
    plt.savefig(f"signals_per_class.png", 
                    format="png",bbox_inches='tight', pad_inches=0.2) 

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:17:37.284685Z","iopub.execute_input":"2023-04-09T14:17:37.285024Z","iopub.status.idle":"2023-04-09T14:19:07.099963Z","shell.execute_reply.started":"2023-04-09T14:17:37.284989Z","shell.execute_reply":"2023-04-09T14:19:07.098972Z"}}
%%time
signals = [' '.join(df_mitbih.iloc[i, :-1].apply(str).values) for i in range(df_mitbih.shape[0])]
y = df_mitbih.iloc[:, -1].values.tolist()
print(len(signals), len(y))

print(f'data has {len(set([sig for line in signals for sig in line.split()]))} out of 16 372 411 unique values.')

# %% [markdown]
# ## Dataset and DataLoader

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.101675Z","iopub.execute_input":"2023-04-09T14:19:07.102251Z","iopub.status.idle":"2023-04-09T14:19:07.116108Z","shell.execute_reply.started":"2023-04-09T14:19:07.102214Z","shell.execute_reply":"2023-04-09T14:19:07.109149Z"}}
class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.FloatTensor([signal.values])                 
        target = torch.LongTensor(np.array(self.df.loc[idx, 'class']))
        return signal, target

    def __len__(self):
        return len(self.df)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.118237Z","iopub.execute_input":"2023-04-09T14:19:07.118727Z","iopub.status.idle":"2023-04-09T14:19:07.126413Z","shell.execute_reply.started":"2023-04-09T14:19:07.118681Z","shell.execute_reply":"2023-04-09T14:19:07.125128Z"}}
def get_dataloader(phase: str, batch_size: int = 96) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        pahse: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=config.seed, stratify=df['label']
    )
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    df = train_df if phase == 'train' else val_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader

# %% [markdown]
# # Models

# %% [markdown]
# ![](https://64.media.tumblr.com/e42e20eb2ec1aea3962c6ace63adf499/70877119c7741403-44/s540x810/c8f722eb2ab3d92c98070554db4815ca8c01510b.png)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.128266Z","iopub.execute_input":"2023-04-09T14:19:07.128749Z","iopub.status.idle":"2023-04-09T14:19:07.351147Z","shell.execute_reply.started":"2023-04-09T14:19:07.128704Z","shell.execute_reply":"2023-04-09T14:19:07.349810Z"}}
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
x = torch.linspace(-10.0, 10.0, 100)
swish = Swish()
swish_out = swish(x)
relu_out = torch.relu(x)

plt.title('Swish function')
plt.plot(x.numpy(), swish_out.numpy(), label='Swish')
plt.plot(x.numpy(), relu_out.numpy(), label='ReLU')
plt.legend();
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.352438Z","iopub.execute_input":"2023-04-09T14:19:07.352815Z","iopub.status.idle":"2023-04-09T14:19:07.367097Z","shell.execute_reply.started":"2023-04-09T14:19:07.352782Z","shell.execute_reply":"2023-04-09T14:19:07.366097Z"}}
class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        norm_type='bachnorm'
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
            
        self.pool = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))   
        
        x = self.pool(x)
        return x

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.368646Z","iopub.execute_input":"2023-04-09T14:19:07.369006Z","iopub.status.idle":"2023-04-09T14:19:07.387302Z","shell.execute_reply.started":"2023-04-09T14:19:07.368969Z","shell.execute_reply":"2023-04-09T14:19:07.386121Z"}}
class CNN(nn.Module):
    def __init__(
        self,
        input_size = 1,
        hid_size = 256,
        kernel_size = 5,
        num_classes = 5,
    ):
        
        super().__init__()
        
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size//4, out_features=num_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)        
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.389260Z","iopub.execute_input":"2023-04-09T14:19:07.389604Z","iopub.status.idle":"2023-04-09T14:19:07.409046Z","shell.execute_reply.started":"2023-04-09T14:19:07.389568Z","shell.execute_reply":"2023-04-09T14:19:07.407350Z"}}
class RNN(nn.Module):
    """RNN module(cell type lstm or gru)"""
    def __init__(
        self,
        input_size,
        hid_size,
        num_rnn_layers=1,
        dropout_p = 0.2,
        bidirectional = False,
        rnn_type = 'lstm',
    ):
        super().__init__()
        
        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            
        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.410738Z","iopub.execute_input":"2023-04-09T14:19:07.411144Z","iopub.status.idle":"2023-04-09T14:19:07.426425Z","shell.execute_reply.started":"2023-04-09T14:19:07.411099Z","shell.execute_reply":"2023-04-09T14:19:07.425170Z"}}
class RNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        n_classes=5,
        kernel_size=5,
    ):
        super().__init__()
            
        self.rnn_layer = RNN(
            input_size=46,#hid_size * 2 if bidirectional else hid_size,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x, _ = self.rnn_layer(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)#.squeeze(1)
        return x

# %% [markdown]
# ### "Attention Mechanism" Quick Reminder
# 
# 
# The attention mechanism is best explained with the example of the seq2seq model, so it would be a great idea to read this interactive [article](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html).
# 
# Models based an architecture such as sequence2sequence, for example, to translate from one language to another, **Attention**, use to clarify the word order, when translating by the decoder, more specificaly to make weights of some words more or less meaningful in the encoder path, for improved translation.
# 
# 
# ### Excerpt from [article](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html):
# At each decoder step, attention:
# + receives attention input: a decoder state ht and all encoder states s1, s2...sk;
# + computes attention scores;
# + For each encoder state sk attention computes its "relevance" for this decoder state ht. Formally, it applies an attention function which receives one decoder state and one encoder state and returns a scalar value **score(ht, sk)**;
# + computes attention weights: a probability distribution - softmax applied to attention scores;
# + computes attention output: the weighted sum of encoder states with attention weights.
#     
# The general computation scheme is shown below.
# <img src="https://lena-voita.github.io/resources/lectures/seq2seq/attention/computation_scheme-min.png" alt="drawing" width="60%" height="60%"/>
# 
# The most popular ways to compute attention scores are:
# 
# + dot-product - the simplest method;
# + bilinear function (aka "Luong attention") - used in the paper Effective Approaches to Attention-based Neural Machine Translation;
# + multi-layer perceptron (aka "Bahdanau attention") - the method proposed in the original paper.
# ![alt](https://lena-voita.github.io/resources/lectures/seq2seq/attention/score_functions-min.png)
# 
# ---
# 
# We will use Attention mechanism in ecg classification, "to clarify" to give more attention to important features, be it features from recurrent layers or convolutional.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.428170Z","iopub.execute_input":"2023-04-09T14:19:07.428645Z","iopub.status.idle":"2023-04-09T14:19:07.446857Z","shell.execute_reply.started":"2023-04-09T14:19:07.428578Z","shell.execute_reply":"2023-04-09T14:19:07.445668Z"}}
class RNNAttentionModel(nn.Module):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        n_classes=5,
        kernel_size=5,
    ):
        super().__init__()
 
        self.rnn_layer = RNN(
            input_size=46,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveMaxPool1d((1))
        self.attn = nn.Linear(hid_size, hid_size, bias=False)
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x_out, hid_states = self.rnn_layer(x)
        x = torch.cat([hid_states[0], hid_states[1]], dim=0).transpose(0, 1)
        x_attn = torch.tanh(self.attn(x))
        x = x_attn.bmm(x_out)
        x = x.transpose(2, 1)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=-1)
        return x

# %% [markdown]
# # Training Stage

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.448322Z","iopub.execute_input":"2023-04-09T14:19:07.448751Z","iopub.status.idle":"2023-04-09T14:19:07.467965Z","shell.execute_reply.started":"2023-04-09T14:19:07.448706Z","shell.execute_reply":"2023-04-09T14:19:07.466919Z"}}
class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))
    
    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x,y)
        self.metrics['f1'] += f1_score(x,y,average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x,y, average='macro', zero_division=1)
        
        self._compute_cm(x, y)
        
    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1
    
    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0
        
    def get_metrics(self):
        return self.metrics
    
    def get_confusion_matrix(self):
        return self.confusion

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.469509Z","iopub.execute_input":"2023-04-09T14:19:07.470091Z","iopub.status.idle":"2023-04-09T14:19:07.491458Z","shell.execute_reply.started":"2023-04-09T14:19:07.470045Z","shell.execute_reply":"2023-04-09T14:19:07.490583Z"}}
class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs):
        self.net = net.to(config.device)
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(phase, batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
    
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(config.device)
            target = target.to(config.device)
            
            output = self.net(data)
            loss = self.criterion(output, target)
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            meter.update(output, target, loss.item())
        
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )
        fig, ax = plt.subplots(figsize=(5, 5))
        cm_ = ax.imshow(confusion_matrix, cmap='hot')
        ax.set_title('Confusion matrix', fontsize=15)
        ax.set_xlabel('Actual', fontsize=13)
        ax.set_ylabel('Predicted', fontsize=13)
        plt.colorbar(cm_)
        plt.show()
        
        return loss
    
    def run(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(phase='train')
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")
            #clear_output()
        

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.492922Z","iopub.execute_input":"2023-04-09T14:19:07.493244Z","iopub.status.idle":"2023-04-09T14:19:07.513416Z","shell.execute_reply.started":"2023-04-09T14:19:07.493212Z","shell.execute_reply":"2023-04-09T14:19:07.512600Z"}}
#model = RNNAttentionModel(1, 64, 'lstm', False)
model = RNNModel(1, 64, 'lstm', True)
#model = CNN(num_classes=5, hid_size=128)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T14:19:07.515981Z","iopub.execute_input":"2023-04-09T14:19:07.516618Z","iopub.status.idle":"2023-04-09T15:11:34.266500Z","shell.execute_reply.started":"2023-04-09T14:19:07.516569Z","shell.execute_reply":"2023-04-09T15:11:34.265299Z"}}
trainer = Trainer(net=model, lr=1e-3, batch_size=96, num_epochs=10)#100)
trainer.run()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:34.268640Z","iopub.execute_input":"2023-04-09T15:11:34.268982Z","iopub.status.idle":"2023-04-09T15:11:34.282767Z","shell.execute_reply.started":"2023-04-09T15:11:34.268949Z","shell.execute_reply":"2023-04-09T15:11:34.281729Z"}}
train_logs = trainer.train_df_logs
train_logs.columns = ["train_"+ colname for colname in train_logs.columns]
val_logs = trainer.val_df_logs
val_logs.columns = ["val_"+ colname for colname in val_logs.columns]

logs = pd.concat([train_logs,val_logs], axis=1)
logs.reset_index(drop=True, inplace=True)
logs = logs.loc[:, [
    'train_loss', 'val_loss', 
    'train_accuracy', 'val_accuracy', 
    'train_f1', 'val_f1',
    'train_precision', 'val_precision',
    'train_recall', 'val_recall']
                                 ]
logs.head()
logs.to_csv('cnn.csv', index=False)

# %% [markdown]
# # Experiments and Results

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:34.284713Z","iopub.execute_input":"2023-04-09T15:11:34.285064Z","iopub.status.idle":"2023-04-09T15:11:34.329872Z","shell.execute_reply.started":"2023-04-09T15:11:34.285031Z","shell.execute_reply":"2023-04-09T15:11:34.328834Z"}}
cnn_model = CNN(num_classes=5, hid_size=128).to(config.device)
cnn_model.load_state_dict(
    torch.load(config.cnn_state_path,
               map_location=config.device)
);
cnn_model.eval();
logs = pd.read_csv(config.cnn_logs)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:11:34.332081Z","iopub.execute_input":"2023-04-09T15:11:34.332594Z","iopub.status.idle":"2023-04-09T15:11:35.496183Z","shell.execute_reply.started":"2023-04-09T15:11:34.332519Z","shell.execute_reply":"2023-04-09T15:11:35.494933Z"}}
colors = ['#C042FF', '#03C576FF', '#FF355A', '#03C5BF', '#96C503', '#C5035B']
palettes = [sns.color_palette(colors, 2),
            sns.color_palette(colors, 4), 
            sns.color_palette(colors[:2]+colors[-2:] + colors[2:-2], 6)]
            
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=logs.iloc[:, :2], palette=palettes[0], markers=True, ax=ax[0], linewidth=2.5,)
ax[0].set_title("Loss Function during Model Training", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)

sns.lineplot(data=logs.iloc[:, 2:6], palette=palettes[1], markers=True, ax=ax[1], linewidth=2.5, legend="full")
ax[1].set_title("Metrics during Model Training", fontsize=15)
ax[1].set_xlabel("Epoch", fontsize=14)

plt.suptitle('CNN Model', fontsize=18)

plt.tight_layout()
fig.savefig("cnn.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("cnn.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:35.497944Z","iopub.execute_input":"2023-04-09T15:11:35.498262Z","iopub.status.idle":"2023-04-09T15:11:35.523992Z","shell.execute_reply.started":"2023-04-09T15:11:35.498231Z","shell.execute_reply":"2023-04-09T15:11:35.522903Z"}}
lstm_model = RNNModel(1, 64, 'lstm', True).to(config.device)
lstm_model.load_state_dict(
    torch.load(config.lstm_state_path,
               map_location=config.device)
);
lstm_model.eval();
logs = pd.read_csv(config.lstm_logs)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:11:35.525560Z","iopub.execute_input":"2023-04-09T15:11:35.525876Z","iopub.status.idle":"2023-04-09T15:11:36.672287Z","shell.execute_reply.started":"2023-04-09T15:11:35.525846Z","shell.execute_reply":"2023-04-09T15:11:36.671079Z"}}
colors = ['#C042FF', '#03C576FF', '#FF355A', '#03C5BF', '#96C503', '#C5035B']
palettes = [sns.color_palette(colors, 2),
            sns.color_palette(colors, 4), 
            sns.color_palette(colors[:2]+colors[-2:] + colors[2:-2], 6)]
            
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=logs.iloc[:, :2], palette=palettes[0], markers=True, ax=ax[0], linewidth=2.5,)
ax[0].set_title("Loss Function during Model Training", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)

sns.lineplot(data=logs.iloc[:, 2:6], palette=palettes[1], markers=True, ax=ax[1], linewidth=2.5, legend="full")
ax[1].set_title("Metrics during Model Training", fontsize=15)
ax[1].set_xlabel("Epoch", fontsize=14)

plt.suptitle('CNN+LSTM Model', fontsize=18)

plt.tight_layout()
fig.savefig("lstm.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("lstm.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:36.673899Z","iopub.execute_input":"2023-04-09T15:11:36.674231Z","iopub.status.idle":"2023-04-09T15:11:36.698966Z","shell.execute_reply.started":"2023-04-09T15:11:36.674197Z","shell.execute_reply":"2023-04-09T15:11:36.697976Z"}}
attn_model = RNNAttentionModel(1, 64, 'lstm', False).to(config.device)
attn_model.load_state_dict(
    torch.load(config.attn_state_path,
               map_location=config.device)
);
attn_model.eval();
logs = pd.read_csv(config.attn_logs)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:11:36.700479Z","iopub.execute_input":"2023-04-09T15:11:36.700839Z","iopub.status.idle":"2023-04-09T15:11:37.871938Z","shell.execute_reply.started":"2023-04-09T15:11:36.700805Z","shell.execute_reply":"2023-04-09T15:11:37.870747Z"}}
colors = ['#C042FF', '#03C576FF', '#FF355A', '#03C5BF', '#96C503', '#C5035B']
palettes = [sns.color_palette(colors, 2),
            sns.color_palette(colors, 4), 
            sns.color_palette(colors[:2]+colors[-2:] + colors[2:-2], 6)]
            
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=logs.iloc[:, :2], palette=palettes[0], markers=True, ax=ax[0], linewidth=2.5,)
ax[0].set_title("Loss Function during Model Training", fontsize=14)
ax[0].set_xlabel("Epoch", fontsize=14)

sns.lineplot(data=logs.iloc[:, 2:6], palette=palettes[1], markers=True, ax=ax[1], linewidth=2.5, legend="full")
ax[1].set_title("Metrics during Model Training", fontsize=15)
ax[1].set_xlabel("Epoch", fontsize=14)

plt.suptitle('CNN+LSTM+Attention Model', fontsize=18)

plt.tight_layout()
fig.savefig("attn.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("attn.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

# %% [markdown]
# ## Experiments and Results for Test Stage

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:37.873220Z","iopub.execute_input":"2023-04-09T15:11:37.873526Z","iopub.status.idle":"2023-04-09T15:11:38.510942Z","shell.execute_reply.started":"2023-04-09T15:11:37.873497Z","shell.execute_reply":"2023-04-09T15:11:38.509681Z"}}
test_df = pd.read_csv(config.test_csv_path)
print(test_df.shape)
test_dataset = ECGDataset(test_df)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=96, num_workers=0, shuffle=False)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:38.517261Z","iopub.execute_input":"2023-04-09T15:11:38.517857Z","iopub.status.idle":"2023-04-09T15:11:38.525750Z","shell.execute_reply.started":"2023-04-09T15:11:38.517805Z","shell.execute_reply":"2023-04-09T15:11:38.524618Z"}}
def make_test_stage(dataloader, model, probs=False):
    cls_predictions = []
    cls_ground_truths = []

    for i, (data, cls_target) in enumerate(dataloader):
        with torch.no_grad():

            data = data.to(config.device)
            cls_target = cls_target.cpu()
            cls_prediction = model(data)
            
            if not probs:
                cls_prediction = torch.argmax(cls_prediction, dim=1)
    
            cls_predictions.append(cls_prediction.detach().cpu())
            cls_ground_truths.append(cls_target)

    predictions_cls = torch.cat(cls_predictions).numpy()
    ground_truths_cls = torch.cat(cls_ground_truths).numpy()
    return predictions_cls, ground_truths_cls

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:38.527734Z","iopub.execute_input":"2023-04-09T15:11:38.528196Z","iopub.status.idle":"2023-04-09T15:11:38.544077Z","shell.execute_reply.started":"2023-04-09T15:11:38.528148Z","shell.execute_reply":"2023-04-09T15:11:38.542841Z"}}
models = [cnn_model, lstm_model, attn_model]


# %% [markdown]
# ### cnn model report

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:11:38.546597Z","iopub.execute_input":"2023-04-09T15:11:38.547271Z","iopub.status.idle":"2023-04-09T15:12:17.929800Z","shell.execute_reply.started":"2023-04-09T15:11:38.547216Z","shell.execute_reply":"2023-04-09T15:12:17.925384Z"}}
y_pred, y_true = make_test_stage(test_dataloader, models[0])
y_pred.shape, y_true.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:12:17.931058Z","iopub.execute_input":"2023-04-09T15:12:17.931509Z","iopub.status.idle":"2023-04-09T15:12:17.964528Z","shell.execute_reply.started":"2023-04-09T15:12:17.931475Z","shell.execute_reply":"2023-04-09T15:12:17.963662Z"}}
report = pd.DataFrame(
    classification_report(
        y_pred,
        y_true,
        output_dict=True
    )
).transpose()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:12:17.965841Z","iopub.execute_input":"2023-04-09T15:12:17.966311Z","iopub.status.idle":"2023-04-09T15:12:18.899925Z","shell.execute_reply.started":"2023-04-09T15:12:17.966277Z","shell.execute_reply":"2023-04-09T15:12:18.898667Z"}}
colors = ['#00FA9A', '#D2B48C', '#FF69B4']#random.choices(list(mcolors.CSS4_COLORS.values()), k = 3)
report_plot = report.apply(lambda x: x*100)
ax = report_plot[["precision", "recall", "f1-score"]].plot(kind='bar',
                                                      figsize=(13, 4), legend=True, fontsize=15, color=colors)

ax.set_xlabel("Estimators", fontsize=15)
ax.set_xticklabels(
    list(id_to_label.values())+["accuracy avg", "marco avg", "weighted avg"],
    rotation=15, fontsize=11)
ax.set_ylabel("Percentage", fontsize=15)
plt.title("CNN Model Classification Report", fontsize=20)

for percentage, p in zip(
    report[['precision', 'recall', 'f1-score']].values,
    ax.patches):
    
    percentage = " ".join([str(round(i*100, 2))+"%" for i in percentage])
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() / 4
    ax.annotate(percentage, (x, y), fontsize=8, rotation=15, fontweight='bold')
fig.savefig("cnn_report.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("cnn_report.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### cnn+lstm model report

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:12:18.901637Z","iopub.execute_input":"2023-04-09T15:12:18.901996Z","iopub.status.idle":"2023-04-09T15:12:53.702612Z","shell.execute_reply.started":"2023-04-09T15:12:18.901929Z","shell.execute_reply":"2023-04-09T15:12:53.701323Z"}}
y_pred, y_true = make_test_stage(test_dataloader, models[1])
y_pred.shape, y_true.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:12:53.704609Z","iopub.execute_input":"2023-04-09T15:12:53.705469Z","iopub.status.idle":"2023-04-09T15:12:53.748313Z","shell.execute_reply.started":"2023-04-09T15:12:53.705409Z","shell.execute_reply":"2023-04-09T15:12:53.747191Z"}}
report = pd.DataFrame(
    classification_report(
        y_pred,
        y_true,
        output_dict=True
    )
).transpose()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:12:53.750230Z","iopub.execute_input":"2023-04-09T15:12:53.750688Z","iopub.status.idle":"2023-04-09T15:12:54.768031Z","shell.execute_reply.started":"2023-04-09T15:12:53.750642Z","shell.execute_reply":"2023-04-09T15:12:54.766403Z"}}
colors = ['#00FA9A', '#D2B48C', '#FF69B4']#random.choices(list(mcolors.CSS4_COLORS.values()), k = 3)
report_plot = report.apply(lambda x: x*100)
ax = report_plot[["precision", "recall", "f1-score"]].plot(kind='bar',
                                                      figsize=(13, 4), legend=True, fontsize=15, color=colors)

ax.set_xlabel("Estimators", fontsize=15)
ax.set_xticklabels(
    list(id_to_label.values())+["accuracy avg", "marco avg", "weighted avg"],
    rotation=15, fontsize=11)
ax.set_ylabel("Percentage", fontsize=15)
plt.title("CNN+LSTM Model Classification Report", fontsize=20)

for percentage, p in zip(
    report[['precision', 'recall', 'f1-score']].values,
    ax.patches):
    
    percentage = " ".join([str(round(i*100, 2))+"%" for i in percentage])
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() / 4
    ax.annotate(percentage, (x, y), fontsize=8, rotation=15, fontweight='bold')
fig.savefig("lstm_report.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("lstm_report.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### cnn+lstm+attention model report

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:12:54.769531Z","iopub.execute_input":"2023-04-09T15:12:54.769899Z","iopub.status.idle":"2023-04-09T15:13:26.337035Z","shell.execute_reply.started":"2023-04-09T15:12:54.769863Z","shell.execute_reply":"2023-04-09T15:13:26.336043Z"}}
y_pred, y_true = make_test_stage(test_dataloader, models[2])
y_pred.shape, y_true.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:13:26.338623Z","iopub.execute_input":"2023-04-09T15:13:26.338945Z","iopub.status.idle":"2023-04-09T15:13:26.371878Z","shell.execute_reply.started":"2023-04-09T15:13:26.338913Z","shell.execute_reply":"2023-04-09T15:13:26.370868Z"}}
report = pd.DataFrame(
    classification_report(
        y_pred,
        y_true,
        output_dict=True
    )
).transpose()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-04-09T15:13:26.373304Z","iopub.execute_input":"2023-04-09T15:13:26.373636Z","iopub.status.idle":"2023-04-09T15:13:27.285357Z","shell.execute_reply.started":"2023-04-09T15:13:26.373604Z","shell.execute_reply":"2023-04-09T15:13:27.284198Z"}}
colors = ['#00FA9A', '#D2B48C', '#FF69B4']#random.choices(list(mcolors.CSS4_COLORS.values()), k = 3)
report_plot = report.apply(lambda x: x*100)
ax = report_plot[["precision", "recall", "f1-score"]].plot(kind='bar',
                                                      figsize=(13, 4), legend=True, fontsize=15, color=colors)

ax.set_xlabel("Estimators", fontsize=15)
ax.set_xticklabels(
    list(id_to_label.values())+["accuracy avg", "marco avg", "weighted avg"],
    rotation=15, fontsize=11)
ax.set_ylabel("Percentage", fontsize=15)
plt.title("CNN+LSTM+Attention Model Classification Report", fontsize=20)

for percentage, p in zip(
    report[['precision', 'recall', 'f1-score']].values,
    ax.patches):
    
    percentage = " ".join([str(round(i*100, 2))+"%" for i in percentage])
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() / 4
    ax.annotate(percentage, (x, y), fontsize=8, rotation=15, fontweight='bold')
fig.savefig("attn_report.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig("attn_report.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Ensemble of all models

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:13:27.287284Z","iopub.execute_input":"2023-04-09T15:13:27.287983Z","iopub.status.idle":"2023-04-09T15:15:12.257269Z","shell.execute_reply.started":"2023-04-09T15:13:27.287933Z","shell.execute_reply":"2023-04-09T15:15:12.256176Z"}}
y_pred = np.zeros((y_pred.shape[0], 5), dtype=np.float32)
for i, model in enumerate(models, 1):
    y_pred_, y_true = make_test_stage(test_dataloader, model, True)
    y_pred += y_pred_
y_pred /= i
y_pred = np.argmax(y_pred, axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:15:12.261919Z","iopub.execute_input":"2023-04-09T15:15:12.262436Z","iopub.status.idle":"2023-04-09T15:15:13.176240Z","shell.execute_reply.started":"2023-04-09T15:15:12.262386Z","shell.execute_reply":"2023-04-09T15:15:13.174780Z"}}
clf_report = classification_report(y_pred, 
                                   y_true,
                                   labels=[0,1,2,3,4],
                                   target_names=list(id_to_label.values()),#['N', 'S', 'V', 'F', 'Q'],
                                   output_dict=True)


plt.figure(figsize=(10, 8))
ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)
plt.title("Ensemble Classification Report", fontsize=20)
plt.savefig(f"ensemble result.svg",format="svg",bbox_inches='tight', pad_inches=0.2)
plt.savefig(f"ensemble result.png", format="png",bbox_inches='tight', pad_inches=0.2)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T15:15:13.177825Z","iopub.execute_input":"2023-04-09T15:15:13.178151Z","iopub.status.idle":"2023-04-09T15:15:13.186978Z","shell.execute_reply.started":"2023-04-09T15:15:13.178120Z","shell.execute_reply":"2023-04-09T15:15:13.185672Z"}}
clf_report

# %% [code]


# %% [code]


# %% [code]
