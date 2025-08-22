import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from tqdm import tqdm

size = 12
FIGSIZE = (4,8/3)
plt.rc('font', size=10)
plt.rc('axes', titlesize=size*1.2)
plt.rc('axes', labelsize=size*1.2)
plt.rc('xtick', labelsize=size*0.9)
plt.rc('ytick', labelsize=size*0.9)
plt.rc('legend', fontsize=size)

def plot_scores(dir):
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig1, ax1 = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    kind = 'perp'
    print(kind)
    data = np.load(os.path.join('models', dir+'_qcd', f'predictions_{kind}.npz'))
    scores = data['scores']
    labels = data['labels']
    upper = np.quantile(scores, 0.995)
    lower = np.quantile(scores, 0.005)
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]

    ax[0].hist(scores[labels==0],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='QCD')
    ax[0].hist(scores[labels==1],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='Top')
    ax[0].set_xlim(lower, upper)
    ax[0].legend()
    ax[0].set_title(f'Trained on QCD')

    meds_qcd = []
    meds_top = []
    for i in range(1, 101):
        nParts = data['nparts'][idx]==i
        meds_qcd.append(np.median(scores[nParts][labels[nParts]==0]))
        meds_top.append(np.median(scores[nParts][labels[nParts]==1]))

    ax1[0].scatter(data['nparts'][idx],scores,
                   c=labels, s=1, alpha=0.1,)
    ax1[0].plot(np.arange(1,101), np.array(meds_qcd), label='QCD')
    ax1[0].plot(np.arange(1,101), np.array(meds_top), label='Top')
    ax1[0].set_ylim(lower, upper)
    ax1[0].set_ylabel('Score')
    ax1[0].set_title(f'Trained on QCD')
    ax1[0].legend(loc='upper right')

    data = np.load(os.path.join('models', dir+'_top', f'predictions_{kind}.npz'))
    scores = data['scores']
    labels = data['labels']
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]
    ax[1].hist(scores[labels==1],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='QCD')
    ax[1].hist(scores[labels==0],
               density=True, bins=50, histtype='step', range=[lower, upper],
               label='Top')
    ax[1].set_xlabel('score')
    ax[1].set_title('Trained on top')

    meds_qcd = []
    meds_top = []
    for i in range(1, 101):
        nParts = data['nparts'][idx]==i
        meds_qcd.append(np.median(scores[nParts][labels[nParts]==1]))
        meds_top.append(np.median(scores[nParts][labels[nParts]==0]))

    ax1[1].scatter(data['nparts'][idx], scores,
                   c=np.abs(labels-1), s=1, alpha=0.1,)
    ax1[1].plot(np.arange(1,101), np.array(meds_qcd))
    ax1[1].plot(np.arange(1,101), np.array(meds_top))
    ax1[1].set_title('Trained on top')
    ax1[1].set_ylim(lower, upper)
    ax1[1].set_ylabel('Score')
    ax1[1].set_xlabel('Number of particles')

    fig.savefig(f'scores_{kind}.png',)
    fig1.savefig(f'nParts_score_{kind}.png',)


def plot_rocs(dir, sic=False):
    fig, ax = plt.subplots(constrained_layout=True)
    score_kinds = ['perp_fixed']
    for score_kind in score_kinds:
        data = np.load(f'models/{dir}_qcd/predictions_{score_kind}.npz')
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, _ = roc_curve(y_true=labels, y_score=-scores)
        num = tpr if sic else 1.
        den = np.sqrt(fpr) if sic else fpr

        ax.plot(tpr, num / den,
            label=f'QCD {roc_auc_score(labels, -scores):.3f}')

        data = np.load(f'models/{dir}_top/predictions_{score_kind}.npz')
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, _ = roc_curve(y_true=labels, y_score=-scores)
        num = tpr if sic else 1.
        den = np.sqrt(fpr) if sic else fpr

        ax.plot(tpr, num / den,
            label=f'Top {roc_auc_score(labels, -scores):.3f}')

    num = np.linspace(0, 1, 1000) if sic else 1.
    den = np.sqrt(np.linspace(0, 1, 1000)) if sic else np.linspace(0, 1, 1000)
    ax.plot(np.linspace(0, 1, 1000), num / den,
        linestyle='--', color='grey')

    ax.grid(which='both')
    ax.legend()
    if not sic:
        ax.set_yscale('log')
        ax.set_ylim(0.9, 1e3)
    ax.set_title(f'20 constituents (all)')
    ax.set_xlabel(r'$\epsilon_S$')
    ax.set_ylabel(r'1 / $\epsilon_B$')
    fig.savefig(f'{"sic" if sic else "roc"}s_fixed_nConst_train.png')


def plot_loss_pC(dir):
    fig, ax = plt.subplots(constrained_layout=True)
    kind = 'perp'
    data = np.load(os.path.join('models', dir+'_qcd', f'predictions_{kind}.npz'))
    losses = data['losses']
    nPos = losses.shape[-1]
    labels = data['labels']
    nparts = data['nparts'].astype(int)
    if ('reverse' in dir) or ('rev' in dir):
        print('reverting scores')
        for i in range(len(losses)):
            losses[i, :nparts[i]-1] = losses[i, :nparts[i]-1][::-1]

    print(np.sum(np.isnan(losses)))
    print(np.sum(losses==0))

    tmp=0
    ax.plot(np.arange(nPos)+1, np.nanmean(losses[labels==tmp], 0), c='red', label='Mean QCD')
    ax.plot(np.arange(nPos)+1, np.nanquantile(losses[labels==tmp], 0.25, 0), alpha=0.5, c='blue')
    ax.plot(np.arange(nPos)+1, np.nanquantile(losses[labels==tmp], 0.75, 0), alpha=0.5, c='blue')
    ax.plot(np.arange(nPos)+1, np.nanmedian(losses[labels==tmp],0), c='blue', label='Median')


    tmp=1
    ax.plot(np.arange(nPos)+1, np.nanmean(losses[labels==tmp], 0), c='red', linestyle='--', label='Mean Top')
    ax.plot(np.arange(nPos)+1, np.nanquantile(losses[labels==tmp], 0.25, 0), alpha=0.5, c='blue', linestyle='--')
    ax.plot(np.arange(nPos)+1, np.nanquantile(losses[labels==tmp], 0.75, 0), alpha=0.5, c='blue', linestyle='--')
    ax.plot(np.arange(nPos)+1, np.nanmedian(losses[labels==tmp],0), c='blue', linestyle='--',)

    ax.set_xlim(0, 100)
    # ax.set_ylim(2.5, 9)
    ax.set_xlabel('Constituent position')
    ax.set_ylabel('Loss (crossentropy)')
    ax.legend(loc='upper right')
    fig.savefig('loss_pC_qcd_reverse_pTsorted.png')
    plt.show()


def plot_probs(dir):
    fig, ax = plt.subplots(constrained_layout=True)
    kind = 'perp'
    data = np.load(os.path.join('models', dir+'_qcd', f'predictions_{kind}.npz'))
    print(list(data.keys()))
    probs = data['probs']

    labels = data['labels']
    nparts = data['nparts'].astype(int)
    if ('reverse' in dir) or ('rev' in dir):
        print('reverting scores')
        for i in range(len(probs)):
            probs[i, :nparts[i]-1] = probs[i, :nparts[i]-1][::-1]

    tmp=1
    mean_probs = np.mean(probs, axis=0)
    for lab, prob in zip(['min', 'max', 'mean', 'median'], mean_probs.T):
        if lab in ['min', 'max']:
            ax.plot(np.arange(102)+1, prob, label=lab)
        if lab == 'mean':
            ax.plot(np.arange(102)+1, prob, label='Random', linestyle='--', c='grey')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(which='both')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 5), minor=True)
    ax.set_xlabel('Constituent position')
    ax.set_ylabel('Probability score')
    fig.savefig('prob_qcd_SE.png')


def plot_max_index(dir):
    fig, ax = plt.subplots(constrained_layout=True)
    kind = 'perp'
    data = np.load(os.path.join('models', dir+'_top', f'predictions_{kind}.npz'))
    print(list(data.keys()))
    probs = data['probs_idx']
    print(probs.shape)
    tmp = data['probs']

    plt.hist(probs[:, :, 1].flatten(), histtype='step', bins=30000)
    # plt.hist(probs[:, i, 0], histtype='step', bins=100)

    plt.yscale('log')
    # colors = plt.cm.jet(np.linspace(0, 1, 9))
    # x = 0
    # for i in np.random.choice(len(probs), 9):
    #     plt.plot(probs[i, :, 0], color=colors[x],)
    #     plt.plot(probs[i, :, 1], color=colors[x], linestyle='--')
    #     x += 1
    plt.savefig('tmp.png')
    exit()

    labels = data['labels']
    nparts = data['nparts'].astype(int)
    if ('reverse' in dir) or ('rev' in dir):
        print('reverting scores')
        for i in range(len(probs)):
            probs[i, :nparts[i]-1] = probs[i, :nparts[i]-1][::-1]

    tmp=1
    mean_probs = np.mean(probs, axis=0)
    for lab, prob in zip(['min', 'max', 'mean', 'median'], mean_probs.T):
        if lab in ['min', 'max']:
            ax.plot(np.arange(102)+1, prob, label=lab)
        if lab == 'mean':
            ax.plot(np.arange(102)+1, prob, label='Random', linestyle='--', c='grey')
    ax.legend()
    ax.grid(which='both')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 5), minor=True)
    ax.set_xlabel('Constituent position')
    ax.set_ylabel('Probability score')
    fig.savefig('idx.png')


def plot_bin_distribution():
    bg_file = "inputFiles/top_benchmark/discretized/train_qcd_pt40_eta30_phi30.h5"
    from preprocess import preprocess_dataframe

    bg_data = pd.read_hdf(bg_file, key="discretized", stop=None)

    bg_data, mask, bg_bins = preprocess_dataframe(
        bg_data, 3, (41, 31, 31), 50, False, False,False, False, False,
    )

    bins = np.zeros(39402)
    for i in tqdm(bg_bins[mask].flatten()):
        bins[i] += 1
    bins /= bins.sum()
    print(bins[bins!=0].min())
    print((bins==0).sum())
    print((bins!=0).sum())

    fig, axes = plt.subplots(ncols=3, figsize=(15, 8), sharey=True, constrained_layout=True)

    max_bin = np.argmax(bins)
    axes[0].plot(bins)
    axes[1].plot(bins)
    axes[2].plot(bins)
    axes[1].set_xlim(max_bin - (41 * 31) / 2, max_bin + 41 * 31 * 1.5)
    axes[2].set_xlim(max_bin - 41 * 3.5, max_bin + 41 * 3.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bin number")
    fig.savefig("figures/bins_dist.png")


if __name__ == '__main__':
    plot_bin_distribution()
    # dirname = '1part/20fixed'
    # dirname = 'N600k_E100_nC100_nBin30_LRcos_reverse'
    # plot_scores(dirname)
    # plot_rocs(dirname, sic=False)
    # plot_rocs(dirname, sic=True)
    # plot_loss_pC(dirname)
    # plot_probs(dirname)
    # plot_max_index(dirname)
