from collections import defaultdict
import math
import torch
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler,BatchSampler
import numpy as np

def update_classwise_accuracies(preds,labels,class_correct,class_totals):
    correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
    for i in range(labels.shape[0]):
        label = labels.data[i].item()
        class_correct[label] += correct[i].item()
        class_totals[label] += 1

def get_accuracies(class_names,class_correct,class_totals):
    accuracy = (100*np.sum(list(class_correct.values()))/np.sum(list(class_totals.values())))
    class_accuracies = [(class_names[i],100.0*(class_correct[i]/class_totals[i])) 
                        for i in class_names.keys() if class_totals[i] > 0]
    return accuracy,class_accuracies

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def split_image_data(train_data,test_data=None,batch_size=20,num_workers=0,
                     valid_size=0.2,sampler=SubsetRandomSampler):
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler(train_idx)
    valid_sampler = sampler(valid_idx)

    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)
    else:
        train_idx, test_idx = train_idx[split:],train_idx[:split]
        train_sampler = sampler(train_idx)
        test_sampler = sampler(test_idx)
        
        test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=test_sampler, num_workers=num_workers)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=valid_sampler, num_workers=num_workers)
    
    return train_loader,valid_loader,test_loader

def calculate_img_stats(dataset):
    imgs_ = torch.stack([img for img,_ in dataset],dim=3)
    imgs_ = imgs_.view(3,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean,imgs_std

def create_csv_from_folder(folder_path,outfile,cols=['id','path']):
    
    f = glob.glob(folder_path+'/*.*')
    
    ids = []
    for elem in f:
        t = elem[elem.rfind('/')+1:]
        ids.append(t[:t.rfind('.')])
    data = {cols[0]:ids,cols[1]:f}    
    df = pd.DataFrame(data,columns=cols)
    df.to_csv(outfile,index=False)