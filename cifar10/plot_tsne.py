import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from einops import rearrange

import pdb
st = pdb.set_trace

c10 = False

M = 10000
N = 1000
np.random.seed(0)

if c10:
    with open('hist_sim2gan_c10_eps=0.3_rec_view_latents1.pkl', 'rb') as f:
        data_c10 = pickle.load(f)
else:
    with open('hist_sim2gan_c100_eps=0.2_rec_view_latents1.pkl', 'rb') as f:
        data_c100 = pickle.load(f)

if not c10:
    data_c10 = data_c100

index2 = np.concatenate([
    np.random.choice(np.arange(M), N, replace=False),
    np.random.choice(np.arange(M), N, replace=False)+M,
    ], axis=0)

tsne = TSNE(n_components=2, perplexity=40, n_iter=300)

dw1 = data_c10['latents'] - data_c10['orig_latents']
dw1 = rearrange(dw1, 'n b m d -> (n b) (m d)')
dw2 = data_c10['gauss_latents'] - data_c10['orig_latents']
dw2 = rearrange(dw2, 'n b m d -> (n b) (m d)')
index = np.random.choice(np.arange(dw1.shape[0]), M, replace=False)

dw = np.concatenate([dw1.numpy()[index], dw2.numpy()[index]], axis=0)

tsne_results_w = tsne.fit_transform(dw)
tsne_results_w = tsne_results_w[index2]

tsne = TSNE(n_components=2, perplexity=20, n_iter=300)

dz1 = data_c10['z_gan'] - data_c10['z_anchor']
dz1 = rearrange(dz1, 'n b d -> (n b) d')
dz2 = data_c10['z_gauss'] - data_c10['z_anchor']
dz2 = rearrange(dz2, 'n b d -> (n b) d')
dz = np.concatenate([dz1.numpy()[index], dz2.numpy()[index]], axis=0)

# N = min(dz1.shape[0], M)
tsne_results_z = tsne.fit_transform(dz)
tsne_results_z = tsne_results_z[index2]

# C10
# data = pd.DataFrame()
sns.set()
sns.set_context('paper')
# fig = plt.figure(figsize=(5,5))

fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)
# fig.suptitle('CIFAR10')

df = pd.DataFrame()
df['t-SNE-2D-x'] = tsne_results_w[:N,0]
df['t-SNE-2D-y'] = tsne_results_w[:N,1]
df['VG'] = 'VG1'
data = df  # data.append(df)

df = pd.DataFrame()
df['t-SNE-2D-x'] = tsne_results_w[N:,0]
df['t-SNE-2D-y'] = tsne_results_w[N:,1]
df['VG'] = 'VG2'
df.index+=len(data)
data = data.append(df)
sns.scatterplot(
    ax=axes[0],
    x="t-SNE-2D-x", y="t-SNE-2D-y",
    hue="VG",
    # palette=sns.color_palette("hls", 10),
    data=data,
    # legend="full",
    alpha=0.2,
)

# plt.title('GAN')
axes[0].set_title('GAN')
# axes[0].set_xlim([0, 8])

df = pd.DataFrame()
df['t-SNE-2D-x'] = tsne_results_z[:N,0]
df['t-SNE-2D-y'] = tsne_results_z[:N,1]
df['VG'] = 'VG1'
data = df  # data.append(df)

df = pd.DataFrame()
df['t-SNE-2D-x'] = tsne_results_z[N:,0]
df['t-SNE-2D-y'] = tsne_results_z[N:,1]
df['VG'] = 'VG2'
df.index+=len(data)
data = data.append(df)
sns.scatterplot(
    ax=axes[1],
    x="t-SNE-2D-x", y="t-SNE-2D-y",
    hue="VG",
    # palette=sns.color_palette("hls", 10),
    data=data,
    # legend="full",
    alpha=0.2,
)

# plt.title('SimCLR')
axes[1].set_title('SimCLR')
# axes[1].set_xlim([0, 0.4])

plt.tight_layout()
fig.savefig('tsne_sns_c10.pdf') if c10 else fig.savefig('tsne_sns_c100.pdf')
fig.savefig('tsne_sns_c10.png') if c10 else fig.savefig('tsne_sns_c100.png')
# plt.show()
