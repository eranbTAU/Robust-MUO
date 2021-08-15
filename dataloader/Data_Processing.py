import os
import pickle
import numpy as np
import torchvision.transforms as transform
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import datetime

def Dim_Redu(data_numpy):
    transforms = transform.Compose(
        [transform.ToTensor(),
        transform.Normalize((0,), (0.5,))])


    features = transforms(data_numpy).squeeze(0).numpy()
    data_tsne = TSNE(n_components=3).fit_transform(features)
    features2 = torch.from_numpy(data_tsne).unsqueeze(0).unsqueeze(1)
    m = nn.AvgPool2d((150, 3), stride=(2, 1), padding=(0,1))
    mm = nn.BatchNorm2d(1)
    output = m(features2)
    output = mm(output)
    output = output.view(-1,3).detach().numpy()
    return output


def get_filepaths(path):
    file_paths = {}
    fb = 0
    for root, directories, files in os.walk(path):
        if fb == 0:
            fb += 1
            fdict = directories
            res_dct = {fdict[i]: i for i in range(0, len(fdict))}
        only_files = []
        for filename in files:
            only_files.append(os.path.join(root, filename))
        if fb != 1:
            file_paths[os.path.basename(root)] = only_files
        fb += 1
    return file_paths, res_dct


def pre_processing(full_file_paths):
    fdict2 = {}
    for cat in full_file_paths.keys():
        Dtrain = []
        for i in full_file_paths[cat]:
            data = pickle.load(file=open(i, "rb"))
            d1, d2 = np.array(data[0])[:,:15], np.array(data[1])[:15]
            data = d1 - d2
            data = Dim_Redu(data)
            Dtrain += data.tolist()
        fdict2[cat] = np.array(Dtrain)
    lengths = [len(v) for v in fdict2.values()][0]
    return fdict2, lengths


if __name__ == '__main__':
    section = '1_section'
    path = ''
    save_path = ''
    full_file_paths, fdict = get_filepaths(path)
    fdict2, lengths = pre_processing(full_file_paths)
    features = []
    labels = []
    for category in fdict2.keys():
        for i in range(lengths):
            data = fdict2[category]
            idx = fdict[category]
            features.append(data[i, :])
            labels.append(idx)

    features, labels = np.array(features).astype(np.float32), np.array(labels).astype(np.int).reshape(-1,1)
    x = datetime.datetime.now()
    path = os.path.join(save_path, 'data_new' + '_' + str(features.shape[0]) + '_' + str(len(fdict2)) + \
                        '_' + x.strftime("%d") + '_' + x.strftime("%m") + '_' + x.strftime("%Y") + \
                        '_' + x.strftime("%H") + '_' + x.strftime("%M") + '_' + section + '.pkl')

    with open(path, 'wb') as H:
        pickle.dump([features, labels], H)






