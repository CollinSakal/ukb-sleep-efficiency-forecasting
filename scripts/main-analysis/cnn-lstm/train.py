import numpy as np
import torch
import torch.distributed as dist
from utils import *
def train_ddp(device, epochs, model, train_loader, valid_loader, test_loader, optimizer, loss_func, save_path, warmup_scheduler=None,
          main_scheduler=None,eval_interval=5,threshold=0.8,warmup_steps=10,type='classification'):
    if device == 0:
        min_val_loss = None; min_val_loss_epoch = 0; max_val_auroc = None; max_val_auroc_epoch = 0; global_loss_list = []; val_loss_list = []; test_loss_list = []
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        train_loader.sampler.set_epoch(epoch)
        model.train()
        if device == 0: loss_list = []
        for step, data in enumerate(train_loader):
            y = data[0]
            x = data[1]
            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x).squeeze(dim=1)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            if device == 0:
                loss_list.append(loss.detach().cpu())

            if device == 0 and step % 50 == 0:
                print(f"Epoch {epoch} Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
                print(f'pred {pred.detach().cpu()[:5]}')
                print(f'y    {y.detach().cpu()[:5]}')
        dist.barrier()

        if device == 0:
            for i in loss_list:
                global_loss_list.append(i.numpy())
            np.savetxt(save_path + 'train_loss.txt', np.array(global_loss_list).flatten())
            print(f"Epoch {epoch} mean loss {torch.mean(torch.Tensor(loss_list))}")

        if epoch < warmup_steps and warmup_scheduler != None:
            warmup_scheduler.step()
        if epoch >= warmup_steps and main_scheduler != None:
            main_scheduler.step()
        if epoch == 0:
            train_loader.dataset.print(save_path+f'/{device}_id.csv')
        if device == 0 and epoch % eval_interval == 0:
            model.eval()
            loss_list.clear()
            preds = []
            ys = []
            with torch.no_grad():
                for step, data in enumerate(valid_loader):
                    y = data[0]
                    x = data[1]
                    x = x.to(device).float()
                    y = y.to(device).float()
                    pred = model(x).squeeze(dim=1)
                    loss = loss_func(pred, y)
                    loss_list.append(loss.detach().cpu().numpy())
                    preds.append(pred.detach().cpu())
                    ys.append(y.detach().cpu())
            val_loss = np.mean(loss_list)
            out_pred = torch.cat(preds,dim=0)
            out_y = torch.cat(ys,dim=0)
            val_loss_list.extend(loss_list)
            np.save(save_path + 'val_loss.npy', np.array(val_loss_list))

            if min_val_loss == None or val_loss < min_val_loss:
                #torch.save(real_model(model), save_path + 'min_val_model')
                min_val_loss = val_loss
                min_val_loss_epoch = epoch

            if type == 'classification':
                result,incorrect=print_accuracy(torch.sigmoid(out_pred.detach()).cpu().ge(0.5).int().numpy(),out_y.type(torch.int).numpy())
                val_auroc,_ = print_auroc(torch.sigmoid(out_pred.detach()).numpy(), out_y.type(torch.int).numpy())
                if np.sum(np.array([result[i] for i in range(2)])>=threshold) == 2:
                    print(f'Epoch {epoch} threshold')
                    acc_str = '_'
                    for acc_index in range(2):
                        acc_str = acc_str + str(int(result[acc_index]*100)) + '_'
                    #torch.save(real_model(model), save_path + str(epoch) + acc_str +'threshold_model')
            elif type == 'regression':
                mae = torch.nn.functional.l1_loss(out_pred,out_y)
                print(f'Epoch {epoch} MAE: {mae}')

            if max_val_auroc == None or val_auroc > max_val_auroc:
                torch.save(real_model(model), save_path + 'max_val_auroc_model')
                max_val_auroc = val_auroc
                max_val_auroc_epoch = epoch

            print(f"Epoch {epoch} Validate mean loss {val_loss}, min loss {min_val_loss}, min epoch {min_val_loss_epoch}, max auroc {max_val_auroc}, max epoch {max_val_auroc_epoch}")

            if test_loader is None:
                continue

            loss_list.clear()
            preds = []
            ys = []
            with torch.no_grad():
                for step, data in enumerate(test_loader):
                    y = data[0]
                    x = data[1]
                    x = x.to(device).float()
                    y = y.to(device).float()
                    pred = model(x).squeeze(dim=1)
                    loss = loss_func(pred, y)
                    loss_list.append(loss.detach().cpu().numpy())
                    preds.append(pred.detach().cpu())
                    ys.append(y.detach().cpu())
            val_loss = np.mean(loss_list)
            out_pred = torch.cat(preds,dim=0)
            out_y = torch.cat(ys,dim=0)
            test_loss_list.extend(loss_list)
            np.save(save_path + 'test_loss.npy', np.array(test_loss_list))

            eval_str = f'Epoch {epoch}, Testset, mean loss {val_loss} '

            if type == 'classification':
                result,incorrect=print_accuracy(torch.sigmoid(out_pred.detach()).cpu().ge(0.5).int().numpy(),out_y.type(torch.int).numpy(),print_str=False)
                auc = print_auroc(torch.sigmoid(out_pred.detach()).numpy(),out_y.type(torch.int).numpy(),print_str=False)
                for i in result:
                    eval_str += f',{i}:{result[i]} '
                eval_str += f', AUROC {auc[0]}, AUPRC {auc[1]}'
            elif type == 'regression':
                mae = torch.nn.functional.l1_loss(out_pred,out_y)
                eval_str += f', MAE: {mae}'

            print(eval_str)
        #torch.save(real_model(model), save_path + 'model')
    return 0