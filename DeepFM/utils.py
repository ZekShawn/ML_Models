import time
import datetime
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def write_log(w):
    file_name = 'data/' + datetime.date.today().strftime('%m%d')+"_{}.log".format("DeepFM")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')


def train_and_eval(model, train_loader, valid_loader, epochs, device, loss_fcn, optimizer, scheduler):
    best_auc = 0.0
    for _ in range(epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, num_fea, label = x[0], x[1], x[2]
            cate_fea, num_fea, label = cate_fea.to(device), num_fea.to(device), label.float().to(device)
            pre = model(cate_fea, num_fea).view(-1)
            loss = loss_fcn(pre, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    _ + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start_time))

        scheduler.step()
        """推断部分"""
        model.eval()
        with torch.no_grad():
            valid_labels, valid_pres = [], []
            for idx, x in tqdm(enumerate(valid_loader)):
                cate_fea, num_fea, label = x[0], x[1], x[2]
                cate_fea, num_fea = cate_fea.to(device), num_fea.to(device)
                pre = model(cate_fea, num_fea).reshape(-1).data.cpu().numpy().tolist()
                valid_pres.extend(pre)
                valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = accuracy_score(valid_labels, valid_pres)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), "data/DeepFM_best.pth")
        write_log('Current AUC: %.6f, Best AUC: %.6f\n' % (cur_auc, best_auc))
