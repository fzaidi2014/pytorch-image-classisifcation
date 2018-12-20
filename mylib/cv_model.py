import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from mylib.utils import *
from mylib.model import *
from mylib.fc import *
import time

class TransferNetworkImg(Network):
    def __init__(self,
                 model_name='DenseNet',
                 lr=0.003,
                 criterion_name ='NLLLoss',
                 optimizer_name = 'Adam',
                 dropout_p=0.2,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_accuracy_file ='best_accuracy.pth',
                 chkpoint_file ='chkpoint_file',
                 head={}):

        
        super().__init__(device=device)
        
        self.model_type = 'transfer'
        
        self.set_transfer_model(model_name,pretrained=pretrained)    
        
        if head is not None:
            self.set_model_head(model_name = model_name,
                                 head = head,
                                 optimizer_name = optimizer_name,
                                 criterion_name = criterion_name,
                                 lr = lr,
                                 dropout_p = dropout_p,
                                 device = device
                                )
            
        self.set_model_params(criterion_name,
                              optimizer_name,
                              lr,
                              dropout_p,
                              model_name,
                              best_accuracy,
                              best_accuracy_file,
                              chkpoint_file,
                              head)
            
        
    def set_model_params(self,criterion_name,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         best_accuracy,
                         best_accuracy_file,
                         chkpoint_file,
                         head):
        
        print('Transfer: best accuracy = {:.3f}'.format(best_accuracy))
        
        super(TransferNetworkImg, self).set_model_params(
                                              criterion_name,
                                              optimizer_name,
                                              lr,
                                              dropout_p,
                                              model_name,
                                              best_accuracy,
                                              best_accuracy_file,
                                              chkpoint_file
                                              )

        self.head = head
        self.model_type = 'transfer'
        self.num_outputs = head['num_outputs']

        if 'class_names' in head.keys():
            self.class_names = head['class_names']
        else:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(head['num_outputs'])))}

        

    def forward(self,x):
        return self.model(x)
        
    def get_model_params(self):
        params = super(TransferNetworkImg, self).get_model_params()
        params['head'] = self.head
        params['model_type'] = self.model_type
        params['device'] = self.device
        return params
    
    def freeze(self,train_classifier=True):
        super(TransferNetworkImg, self).freeze()
        if train_classifier:
            if self.model_name.lower() == 'densenet':
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif self.model_name.lower() == 'resnet34':
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            
                
    def set_transfer_model(self,mname,pretrained=True):   
        self.model = None
        if mname.lower() == 'densenet':
            self.model = models.densenet121(pretrained=pretrained)
            
        elif mname.lower() == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)

        elif mname.lower() == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)

        if self.model is not None:
            print('set_transfer_model: self.Model set to {}'.format(mname))
        else:
            print('set_transfer_model:Model {} not supported'.format(mname))
            
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head = {'num_inputs':128,
                                'num_outputs':10,
                                'layers':[],
                                'class_names':{}
                               },
                         optimizer_name = 'Adam',
                         criterion_name = 'NLLLoss',
                         lr = 0.003,
                         dropout_p = 0.2,
                         device = None):
        
        
        if model_name.lower() == 'densenet':
            if hasattr(self.model,'classifier'):
                in_features =  self.model.classifier.in_features
            else:
                in_features = self.model.classifier.num_inputs
            self.model.classifier = FC(num_inputs=in_features,
                                       num_outputs=head['num_outputs'],
                                       layers = head['layers'],
                                       class_names = head['class_names'],
                                       non_linearity = head['non_linearity'],
                                       model_type = head['model_type'],
                                       model_name = head['model_name'],
                                       dropout_p = dropout_p,
                                       optimizer_name = optimizer_name,
                                       lr = lr,
                                       criterion_name = criterion_name,
                                       device=device
                                      )
            
        elif model_name.lower() == 'resnet50' or model_name.lower() == 'resnet34':
            if hasattr(self.model,'fc'):
                in_features =  self.model.fc.in_features
            else:
                in_features = self.model.fc.num_inputs

            self.model.fc = FC(num_inputs=in_features,
                               num_outputs=head['num_outputs'],
                               layers = head['layers'],
                               class_names = head['class_names'],
                               non_linearity = head['non_linearity'],
                               model_type = head['model_type'],
                               model_name = head['model_name'],
                               dropout_p = dropout_p,
                               optimizer_name = optimizer_name,
                               lr = lr,
                               criterion_name = criterion_name,
                               device=device
                              )
         
        self.head = head
        
        print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          in_features,
                                                                          head['layers'],
                                                                          head['num_outputs']))
    
    def _get_dropout(self):
        if self.model_name.lower() == 'densenet':
            return self.model.classifier._get_dropout()
        
        elif self.model_name.lower() == 'resnet50' or self.model_name.lower() == 'resnet34':
            return self.model.fc._get_dropout()
        
            
    def _set_dropout(self,p=0.2):
        
        if self.model_name.lower() == 'densenet':
            if self.model.classifier is not None:
                print('DenseNet: setting head (FC) dropout prob to {:.3f}'.format(p))
                self.model.classifier._set_dropout(p=p)
                
        elif self.model_name.lower() == 'resnet50' or self.model_name.lower() == 'resnet34':
            if self.model.fc is not None:
                print('ResNet: setting head (FC) dropout prob to {:.3f}'.format(p))
                self.model.fc._set_dropout(p=p)
        


