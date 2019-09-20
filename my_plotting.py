
import numpy as np
from matplotlib import pyplot as plt


def plt_acc_colored(all_scores, ds_name, boost_names, noise_level, id):
    # combine the scores for box-plots
    all_scores = np.transpose(np.array([np.array(sc) for sc in all_scores]))

    # Create box plot for the three
    fig, ax = plt.subplots(figsize=(60, 30))
    plt.xlim(0, len(boost_names))
    ax.set_title('Cross-Validation Performance of boosting models on ' + ds_name + ' Dataset', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.xticks(range(1, len(boost_names) + 1), boost_names)


    k = 2.5
    while k < len(boost_names):
        plt.axvline(k)
        k = k + 10


    box_plt = ax.boxplot(all_scores, patch_artist=True)
    for i, patch in enumerate(box_plt["boxes"]):
        if i < 2:
            continue
        if i % 2 == 0:
            try:
                patch.set_facecolor("lightblue")
            except:
                pass
        else:
            try:
                patch.set_facecolor("lightgreen")
            except:
                pass

    ax.set_xticklabels(boost_names, rotation=90, fontsize=23)
    # ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=24)
    # plt.rc('xtick', labelsize=27)
    # plt.rc('ytick', labelsize=27)

    plt.savefig("acc{}-n{:.2f}.png".format(id, noise_level))
    plt.clf()













def plt_acc(all_scores, ds_name, boost_names, noise_level, id):
    # combine the scores for box-plots
    all_scores = np.transpose(np.array([np.array(sc) for sc in all_scores]))

    # Create box plot for the three
    fig, ax = plt.subplots(figsize=(50, 30))
    ax.set_title('Cross-Validation Performance of boosting models on ' + ds_name + ' Dataset', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=20)

    ax.boxplot(all_scores, labels=boost_names)
    ax.set_xticklabels(boost_names, rotation=90, fontsize=18)
    plt.savefig("acc{}-n{:.2f}.png".format(id, noise_level))




def plt_scores(times, boost_names, noise_level, id):

    # plot time taken
    times = np.array(times)

    fig, ax = plt.subplots(figsize=(30, 30))
    index = np.arange(times.size)

    # Create box plot for the three
    ax.set_ylabel('Time (seconds)', fontsize=16)
    ax.set_title('Times taken by the boosting models', fontsize=20)

    ax.bar(index, times, alpha=0.4, color='b', label='Times')

    ax.set_xticks(index)
    ax.set_xticklabels(boost_names, rotation=45, fontsize=14)

    plt.savefig("figs/time{}-n{:.2f}.png".format(id, noise_level))
