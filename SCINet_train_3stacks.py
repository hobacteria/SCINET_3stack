import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.SCINet_3stacks import SCINet
from models.SCINet_decompose import SCINet_decompose



def process_one_batch_SCINet(dataset_object, batch_x, batch_y, model):
    batch_x = batch_x.double()
    batch_y = batch_y.double()

    if model.stacks  == 1:
        outputs = model(batch_x)
    elif model.stacks  == 2:
        outputs, mid = model(batch_x)
    elif model.stacks  == 3:
        outputs, mid_1, mid_2 = model(batch_x)
    else:
        print('Error!')

    if dataset_object.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
    if model.stacks == 2:
        mid_scaled = dataset_object.inverse_transform(mid)
    if model.stacks == 3:
        mid_1_scaled = dataset_object.inverse_transform(mid_1)
        mid_2_scaled = dataset_object.inverse_transform(mid_2)
        
    f_dim = 0
    
    batch_y = batch_y[-model.output_len:,f_dim:]
    batch_y_scaled = dataset_object.inverse_transform(batch_y)

    if model.stacks  == 1:
        return outputs, outputs_scaled, 0,0,0,0, batch_y, batch_y_scaled
    elif model.stacks  == 2:
        return outputs, outputs_scaled, mid, mid_scaled,0,0 ,batch_y, batch_y_scaled
    
    elif model.stacks  == 3:
        return outputs, outputs_scaled, mid_1, mid_1_scaled,mid_2,mid_2_scaled, batch_y, batch_y_scaled
    
    else:
        print('Error!')
            
def valid(valid_data, valid_loader, criterion,model):
    model.eval()
    total_loss = []

    preds = []
    trues = []
    mids = []
    mids_2 = []
    pred_scales = []
    true_scales = []
    mid_scales = []
    mid_2_scales = []

    for i, (batch_x, batch_y) in enumerate(valid_loader):
        pred, pred_scale, mid, mid_scale,mid_2, mid_2_scale, true, true_scale = process_one_batch_SCINet(
            valid_data, batch_x, batch_y,model)
        if model.stacks  == 1:
            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        elif model.stacks  == 2:
            loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            mids.append(mid.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            mid_scales.append(mid_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())
            
        elif model.stacks  == 3:
            loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            mids.append(mid.detach().cpu().numpy())
            mids_2.append(mid_2.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            mid_scales.append(mid_scale.detach().cpu().numpy())
            mid_2_scales.append(mid_2_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        else:
            print('Error!')

        total_loss.append(loss)
    total_loss = np.average(total_loss)
    

    if model.stacks  == 1:
        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1)
        trues = trues.reshape(-1)
        true_scales = true_scales.reshape(-1)
        pred_scales = pred_scales.reshape(-1)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
    
    elif model.stacks == 2:
        preds = np.array(preds)
        trues = np.array(trues)
        mids = np.array(mids)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)
        mid_scales = np.array(mid_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
        mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
        # print('test shape:', preds.shape, mids.shape, trues.shape)
        ## metric.py
        mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
        print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))   

   
   
   
    elif model.stacks == 3:
        preds = np.array(preds)
        trues = np.array(trues)
        mids = np.array(mids)
        mids_2 = np.array(mids_2)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)
        mid_scales = np.array(mid_scales)
        mid_2_scales = np.array(mid_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
        mids_2 = mids_2.reshape(-1, mids_2.shape[-2], mids_2.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
        mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
        mid_2_scales = mid_2_scales.reshape(-1, mid_2_scales.shape[-2], mid_2_scales.shape[-1])
        # print('test shape:', preds.shape, mids.shape, trues.shape)
        # print('test shape:', preds.shape, mids.shape, trues.shape)
        ## metric.py
        mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
        print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        mae, mse, rmse, mape, mspe, corr = metric(mids_2, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(mid_2_scales, true_scales)
        print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
       
        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
    else:
        print('Error!')

    return total_loss
            
def train(model,train_data,train_loader,valid_data,valid_loader,model_name):
    path = 'C:\\Users\\ghrbs\\daicon_bycycle\\saved_models_with_logs'
    if not os.path.exists(path):
        os.makedirs(path)
    time_now = time.time()
    
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(15, verbose=True)
    lr = 0.005
    writer = SummaryWriter('event/run_ETTh/{}'.format(model_name))
    
    model_optim = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    # if self.args.use_amp:
    #     scaler = torch.cuda.amp.GradScaler()

    # if self.args.resume:
    #     self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        
    
    epoch_start = 0
    train_epochs = 10000
    for epoch in range(epoch_start, train_epochs):
        
        iter_count = 0
        train_loss = []
        
        model.train()
        epoch_time = time.time()
        for i, (batch_x,batch_y) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            pred, pred_scale, mid, mid_scale,mid_2, mid_2_scale, true, true_scale = process_one_batch_SCINet(
                train_data, batch_x, batch_y,model)

            if model.stacks == 1:
                loss = criterion(pred, true)
            elif model.stacks == 2:
                loss = criterion(pred, true) + criterion(mid, true)
            elif model.stacks == 3:
                loss = criterion(pred, true) + criterion(mid, true) + criterion(mid_2, true)
            else:
                print('Error!')

            train_loss.append(loss.item())
            
            if (i+1) % 100==0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time()-time_now)/iter_count
                left_time = speed*((train_epochs - epoch)*train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            # if self.args.use_amp:
            #     print('use amp')    
            #     scaler.scale(loss).backward()
            #     scaler.step(model_optim)
            #     scaler.update()
            loss.backward()
            model_optim.step()
        
        print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
        train_loss = np.average(train_loss)
        print('--------start to validate-----------')
        valid_loss = valid(valid_data, valid_loader, criterion,model)

        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('valid_loss', valid_loss, global_step=epoch)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, valid_loss))

        # writer.add_scalar('train_loss', train_loss, global_step=epoch)
        # writer.add_scalar('valid_loss', valid_loss, global_step=epoch)

        early_stopping(valid_loss, model, path)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #lr = adjust_learning_rate(model_optim, epoch+1, lr, lradj = 2)
        
    save_model(epoch, lr, model, path, model_name='SCINet', horizon=model.output_len)
    best_model_path = path+'/'+'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    return model