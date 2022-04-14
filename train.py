opt={'base_lr':1e-4}
opt['reduce_lr_by'] = 0.1 # Reduce learning rate by 10 times
opt['atWhichReduce'] = [500000] # Reduce learning rate at these iterations.
opt['batch_size'] = 8 #Enter batch size
opt['atWhichSave'] = [2,10002,20002,30002,40002,50002,60002,70002,80002,90002,100002]
opt['iterations'] = 100002
dry_run_iterations = 100
dry_run = False
metric_average_file = 'metric_average.txt'
test_amplification_file = 'test_amplification.txt'
train_amplification_file = 'train_amplification.txt'
# These are folders
save_weights = 'weights'
save_images = 'images' # make this change below also
save_csv_files = 'csv_files'

import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob

from common_classes import load_data, run_test
from network import Net
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

shutil.rmtree(metric_average_file, ignore_errors = True)
shutil.rmtree(test_amplification_file, ignore_errors = True)
shutil.rmtree(train_amplification_file, ignore_errors = True)

shutil.rmtree(save_weights, ignore_errors = True)
shutil.rmtree(save_images, ignore_errors = True)
shutil.rmtree(save_csv_files, ignore_errors = True)

os.makedirs(save_weights)
os.makedirs(save_images)
os.makedirs(save_csv_files)

train_files = glob.glob('./short_train_validation/0*_00_0.1s.ARW')
train_files +=glob.glob('./short_train_validation/2*_00_0.1s.ARW')
if dry_run:
    train_files = train_files[:2]
    opt['iterations'] = dry_run_iterations

# Select random fraction of data
data_fraction = 0.4
data_amount = int(np.floor(len(train_files) * data_fraction))
print(f"Using {data_amount} photos ({data_fraction*100:.2f}% from {len(train_files)}")
train_files = random.sample(train_files, data_amount)
    
gt_files = []
for x in train_files:
    gt_files += glob.glob('./long_train_validation/*'+x[-17:-12]+'*.ARW')
print(f"training with {len(train_files)} photos")    
print(f"Evaluating training each time with {len(gt_files)} photos")
dataloader_train = DataLoader(load_data(train_files,gt_files,train_amplification_file,20,gt_amp=True,training=True), batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

test_files = glob.glob('./short_testing/1*_00_0.1s.ARW') 
if dry_run:
    test_files = test_files[:2]
    
gt_files = []
for x in test_files:
    gt_files = gt_files+ glob.glob('./long_testing/*'+x[-17:-12]+'*.ARW')
dataloader_test = DataLoader(load_data(test_files,gt_files,test_amplification_file,2,gt_amp=True,training=False), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

for i,img in enumerate(dataloader_train):    
    print('Input image size : {}, GT image size : {}'.format(img[0].size(), img[1].size()))    
    break
    
############ Training Begins

device = torch.device("cuda")
model = Net()
print(model)
print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

iter_num = 0
l1_loss = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
optimizer.zero_grad()
loss_list = ['L1_loss']
loss_iter_list = ['Iteration']
iter_LR = ['Iter_LR']
            

while iter_num<opt['iterations']:
    for _, img in enumerate(dataloader_train):
        low = img[0].to(device)
        gt = img[1].to(device)
        model.train()
        pred = model(low)
        iter_num +=1
        #loss2 = feature_loss(pred, gt, which='relu2')
        loss1 = l1_loss(pred,gt)
        #print('l1,vgg loss: ',loss1.item(), loss2.item())
        loss = loss1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        
        if iter_num>opt['iterations']:
            break        
        
        if iter_num%10==0:
            print(iter_num)
            if iter_num%100==0:
                loss_list.append('{},{},{}'.format(loss1.item(),-1,-1))
                loss_iter_list.append(iter_num)
                iter_LR.append(optimizer.param_groups[0]['lr'])
                
        if iter_num in opt['atWhichSave']:
            print('testing......')
            if iter_num == opt['atWhichSave'][0]:
                mode = 'w'
            else:
                mode = 'a'
            run_test(model, dataloader_test, iter_num, save_images, save_csv_files, metric_average_file, mode, training=True)
            torch.save({'model': model.state_dict()},os.path.join(save_weights,'weights_{}'.format(iter_num)))

np.savetxt(os.path.join(save_csv_files,'loss_curve.csv'),[p for p in zip(loss_iter_list,loss_list,iter_LR)],delimiter=',',fmt='%s')

   
                