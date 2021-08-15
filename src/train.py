from process_load.RoboDataLoader import Get_Loader
from torchvision.transforms import transforms
from tqdm.auto import tqdm as tq
from net.utilis import Save_Network_train, Save_Results, plot_confusion_matrix, \
    Save_ResultsNet, Load_Network, multi_acc, axplot, plot_the_test, weights_init_uniform_rule
from net.NetFlip_new import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import logging
from torchsummary import summary

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

num = '2'
num2 = '2'
completeName2 = os.path.join(print_path, namename)
loggar = logging.getLogger()
loggar.setLevel(logging.DEBUG)
fh = logging.FileHandler(completeName2)
fh.setLevel(logging.DEBUG)
loggar.addHandler(fh)

# init_save
Parallel = False
init_weights = True
ph = '.pth'
nph = '.npy'
net_name = 'train_NN_'
net_name_err = 'train_NN_err'
net_name2 = 'train_NN2_'
best_name = 'best_NN_'
val_name = 'val_NN_loss'
val_loss = 'val_NN_loss'
val_acc = 'val_NN_acc'
test_loss = 'test_NN_loss'
test_acc = 'test_NN_acc'
score_net_PATH = 
score_plot_PATH = 

# save
save_net_path = Save_ResultsNet(score_net_PATH, net_name_err, num, ph)
save_net_path2 = Save_ResultsNet(score_net_PATH, net_name, num2, ph)
save_trainloss_path = Save_Results(score_plot_PATH, net_name, num, nph)
save_trainloss_path2 = Save_Results(score_plot_PATH, net_name2, num, nph)
save_trainloss_best_path = Save_Results(score_plot_PATH, best_name, num, nph)
save_valloss_path = Save_Results(score_plot_PATH, val_loss, num, nph)
save_valacc_path = Save_Results(score_plot_PATH, val_acc, num, nph)
save_testloss_path = Save_Results(score_plot_PATH, test_loss, num, nph)
save_testacc_path = Save_Results(score_plot_PATH, test_acc, num, nph)
# load
# load_path = save_net_path

# dir pkl
path = 
root = 

test_path = 
test_root = 

# load_path = 
# Cuda
ngpu = torch.cuda.device_count()
if Parallel:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #
    CUDA_VISIBLE_DEVICES=0 #
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") #
else:
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") #


# Hyperparameters
dtype = torch.cuda.FloatTensor
indtype = torch.cuda.LongTensor
batch_size = 1
lr = 0.00014204445519960077 
# lr = 1e-3 #1e-3 10 epoch
beta1, beta2 = 0.9343523686735326, 0.999
n_epochs = 30 

# dataloader
TransForms = transforms.Compose(
    [transforms.ToTensor()])

trainloader = Get_Loader(root, path, split='Train and validation', transform=TransForms,
                         val=False, batch_size=batch_size, shuffle=True, num_workers=0)

valloader = Get_Loader(root, path, split='Train and validation', transform=TransForms,
                         val=True, batch_size=batch_size, shuffle=True, num_workers=0)

testloader = Get_Loader(test_root, test_path, split='Test', transform=TransForms,
                         val=False, batch_size=batch_size, shuffle=True, num_workers=0)

classes_name = ('Book&Notebook', 'Bottle', 'Cellphone', 'Fork', 'Hammer', 'Mug',
           'Plate', 'Ruler', 'Scissors', 'Screwdriver', 'Spoon')

# net
net = Flip_UNet2()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if Parallel:
    net = nn.DataParallel(net)
    net.to(device)
else:
    net.to(device)

if init_weights:
    net.apply(weights_init_uniform_rule)
    # Load_Network(load_path, net, optimizer=None)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=3e-2) #3e-2
criterion = torch.nn.CrossEntropyLoss().type(dtype)
scheduler = ReduceLROnPlateau(optimizer, 'min')


################################### train ###################################
if __name__ == '__main__':
    try:
        start_time = time.time()
        epoch_loss = []
        epoch_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        print_loss = 0.0
        print_Valloss = 0.0
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            running_acc = 0.0
            net.train()
            bar = tq(trainloader, postfix={"Epoch": epoch, "Train_loss": 0.0, "Train_Acc": 0.0,
                                           "Batch_Size": batch_size, "Learning Rate": lr})
            for idx, (features, labels) in enumerate(bar):
                features = features.squeeze(3).type(dtype).detach()
                labels = labels.squeeze(1).type(indtype).detach()
                net.zero_grad()
                pred = net(features)
                loss = criterion(pred, labels)
                train_acc = multi_acc(pred, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_acc += train_acc.item()
                bar.set_postfix(ordered_dict={"Epoch": epoch, "Train_loss": loss.item(), "Train_Acc": train_acc.item(),
                                              "Batch_Size": batch_size, "Learning Rate": optimizer.param_groups[0]['lr']})

                d = {"Epoch": epoch, "Train_loss": loss.item(), "Train_Acc": train_acc.item(),
                     "Batch_Size": batch_size, "Learning Rate": optimizer.param_groups[0]['lr']}
                loggar.debug(d)

            ################################### validation ####################################
            running_val_acc = 0.0
            running_val_loss = 0.0
            val_bar = tq(valloader, postfix={"Epoch": epoch, "Val_loss": 0.0, "Val_Acc": 0.0,
                                           "Batch_Size": batch_size, "Learning Rate": lr})
            net.eval()
            with torch.no_grad():
                all_preds = torch.tensor([]).to(device=device)
                all_label = torch.tensor([]).to(device=device)
                for i, (features_val, labels_val) in enumerate(val_bar):
                    features_val = features_val.squeeze(3).to(device=device, dtype=torch.float)
                    labels_val = labels_val.squeeze(1).to(device=device, dtype=torch.long)
                    out_val = net(features_val)
                    val_loss = criterion(out_val, labels_val)
                    val_acc = multi_acc(out_val, labels_val)
                    _, predicted = torch.max(out_val, 1)
                    all_preds = torch.cat((all_preds, predicted), dim=0)
                    all_label = torch.cat((all_label, labels_val), dim=0)
                    running_val_loss += val_loss.item()
                    running_val_acc += val_acc.item()

                    val_bar.set_postfix(ordered_dict={"Epoch": epoch, "Val_Loss": val_loss.item(), "Val_Acc": val_acc.item(),
                                                      "Batch_Size": batch_size, "Learning Rate": optimizer.param_groups[0]['lr']})

                    dv = {"Epoch": epoch, "Val_Loss": val_loss.item(), "Val_Acc": val_acc.item(),
                         "Batch_Size": batch_size, "Learning Rate": optimizer.param_groups[0]['lr']}
                    loggar.debug(dv)

            epoch_loss.append(running_loss / len(trainloader))
            epoch_val_loss.append(running_val_loss / len(valloader))
            epoch_acc.append(running_acc / len(trainloader))
            epoch_val_acc.append(running_val_acc / len(valloader))
            scheduler.step(loss)

    except:
        print("Memory error.")
        Save_Network_train(save_net_path, epoch, net, optimizer)

    Save_Network_train(save_net_path2, epoch, net, optimizer)
    axplot(epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc, file_name=net_name+num, save_plot=score_plot_PATH)
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--- Training + Validation Time --- : {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    ################################### test ####################################
    correct = 0
    total = 0
    class_correct = list(0. for i in range(11))
    class_total = list(0. for i in range(11))
    test_losses = []
    test_acc = []
    net.eval()
    with torch.no_grad():
        all_preds_test = torch.tensor([]).to(device=device)
        all_label_test = torch.tensor([]).to(device=device)
        for epoc, (features_test, labels_test) in enumerate(testloader):
            features_test = features_test.squeeze(3).to(device=device, dtype=torch.float)
            labels_test = labels_test.squeeze(1).to(device=device, dtype=torch.long)
            outputs = net(features_test)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels_test).squeeze()

            all_preds_test = torch.cat((all_preds_test, predicted), dim=0)
            all_label_test = torch.cat((all_label_test, labels_test), dim=0)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

            T_loss = criterion(outputs, labels_test)
            acc_test = multi_acc(outputs, labels_test)
            test_losses.append(T_loss.cpu().numpy())
            test_acc.append(acc_test.item())

            dt = {"Epoch": epoc, "Test_Loss": T_loss.item(), "Test_Acc": acc_test.item(),
                  "Batch_Size": batch_size, "Learning Rate": optimizer.param_groups[0]['lr']}
            loggar.debug(dt)

            for i in range(len(labels_test)):
                label = labels_test[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the test set: %d %%' % (
            100 * correct / total))

    for i in range(11):
        print('Accuracy of %5s : %2d %%' % (
            classes_name[i], 100 * class_correct[i] / class_total[i]))

    plot_the_test(test_losses, test_acc, file_name='test' + '_' + net_name + num, save_plot=score_plot_PATH)
    plot_confusion_matrix(all_label_test, all_preds_test,
                          classes=classes_name,
                          file_name='confusion_matrix_test' + '_' + net_name + num,
                          save_plot=score_plot_PATH,
                          normalize=True,
                          title='Confusion Matrix Test')

