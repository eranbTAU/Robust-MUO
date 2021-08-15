from torch import nn, optim
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
import torch
import time
from datetime import date
import matplotlib.pyplot as plt
from FMG_Project.net.NetFlip_new import Autoencoder
from FMG_Project.net.utilis import Save_Network, Load_Network, DataBuilder, customLoss, weights_init_uniform_rule, save_pickle


today = date.today()

def Dim_Redu_VAR(device, data, cat, save_net_path, save_plot, train_net, test_net, plot=True):
    data_set = DataBuilder(data.astype(np.float32), device)
    trainloader = DataLoader(dataset=data_set, batch_size=256)
    D_in = data_set.x.shape[1]
    H = 24
    H2 = 16
    num_class = 11
    beta1, beta2 = 0.9343523686735326, 0.999
    # model = Autoencoder(D_in, H, H2).to(device)
    model = Autoencoder(D_in, H, H2, num_class)
    # model = nn.DataParallel(model)
    model.to(device)
    if train_net:
        model.apply(weights_init_uniform_rule)

    lr = 0.027012269635149896
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4)

    if test_net:
        load_path = save_net_path + '/'+ cat +'.pth'
        Load_Network(load_path, model, optimizer)

    if train_net:
        loss_mse = customLoss()
        epochs = 1  
        train_losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(trainloader):
                # data = data.type(dtype)
                data = data.to(device=device, dtype=torch.float)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_mse(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if epoch % 1 == 0:
                print('==============================> Epoch: {} Average loss: {:.4f} lr: {}'.format(
                    epoch, train_loss / len(trainloader.dataset), optimizer.param_groups[0]['lr']))

                train_losses.append(train_loss / len(trainloader.dataset))

        Save_Network(save_net_path, epoch, model, optimizer, cat)

    model.eval()
    mu_output = []
    logvar_output = []
    with torch.no_grad():
        for i, (data) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mu_output.append(mu)
            mu_result = torch.cat(mu_output, dim=0)
            logvar_output.append(logvar)

    if plot:
        train_plot(mu_result, cat, save_plot)

    return recon_batch, mu_result, logvar

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

def pre_processing(device, full_file_paths, save_net_path, save_plot):                       
    fdict2 = {}
    for cat in full_file_paths.keys():
        Dtrain = []
        for i in full_file_paths[cat]:
            data = pickle.load(file=open(i, "rb"))
            d1, d2 = np.array(data[0])[:,:15], np.array(data[1])[:15]
            data = d1 - d2
            Dtrain += data.tolist()
        fdict2[cat] = np.array(Dtrain)
        start_plot(fdict2[cat], cat, save_plot)
        recon_batch, mu_result, logvar = Dim_Redu_VAR(device, fdict2[cat], cat, save_net_path, save_plot, train_net=False, test_net=True, plot=True)
        dat_mu = mu_result.cpu().numpy().astype(np.float32)
        fdict2[cat] = dat_mu
    lengths = [len(v) for v in fdict2.values()][0]
    return fdict2, lengths

if __name__ == '__main__':
    start_time = time.time()
    ngpu = torch.cuda.device_count()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES = 0
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dtype = torch.cuda.FloatTensor
    num = 5
    section = 'Data_Process_test_VAR' + str(num)
    full_file_paths, fdict = get_filepaths(path)
    fdict2, lengths = pre_processing(device, full_file_paths, save_net_path, save_plot)
    features = []
    labels = []
    for category in fdict2.keys():
        for i in range(lengths):
            data = fdict2[category]
            idx = fdict[category]
            features.append(data[i, :])
            labels.append(idx)


    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--- Training Time --- : {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    save_pickle(features, labels, save_path, fdict2, section)
