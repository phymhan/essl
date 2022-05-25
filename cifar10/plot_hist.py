import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

import pdb
st = pdb.set_trace

with open('hist_sim2gan_c10_eps=0.3_rec_stats.pkl', 'rb') as f:
    data_c10 = pickle.load(f)

with open('hist_sim2gan_c100_eps=0.2_rec_stats.pkl', 'rb') as f:
    data_c100 = pickle.load(f)

c10 = False

if not c10:
    data_c10 = data_c100
# data_c10 = data_c100

x_label = 'L2 Norm'

# C10
# data = pd.DataFrame()
sns.set()
sns.set_context('paper')
# fig = plt.figure(figsize=(5,5))

fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)
# fig.suptitle('CIFAR10')

df = pd.DataFrame()
df[x_label] = data_c10['g_norms']
df['VG'] = 'VG1'
data = df  # data.append(df)

df = pd.DataFrame()
df[x_label] = data_c10['w_norms']
df['VG'] = 'VG2'
df.index+=len(data)
data = data.append(df)

sns.histplot(ax=axes[0], data=data, x=x_label, hue='VG', kde=False, bins=np.linspace(0,8,16), stat="density", common_norm=False)
# plt.title('GAN')
axes[0].set_title('GAN')
# axes[0].set_xlim([0, 8])

df = pd.DataFrame()
df[x_label] = data_c10['f_norms']
df['VG'] = 'VG1'
data = df  # data.append(df)

df = pd.DataFrame()
df[x_label] = data_c10['z_norms']
df['VG'] = 'VG2'
df.index+=len(data)
data = data.append(df)

# sns.set()
# sns.set_context('paper')
# fig = plt.figure(figsize=(5,5))

sns.histplot(ax=axes[1], data=data, x=x_label, hue='VG', kde=False, bins=np.linspace(0,0.4,16), stat="density", common_norm=False)
# plt.title('SimCLR')
axes[1].set_title('SimCLR')
# axes[1].set_xlim([0, 0.4])

plt.tight_layout()
fig.savefig('hist_sns_c10.pdf') if c10 else fig.savefig('hist_sns_c100.pdf')
# plt.show()
