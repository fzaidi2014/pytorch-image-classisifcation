import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import defaultdict
from mylib.utils import *



class Network(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,x):
        pass
    
    def train_(self,trainloader,criterion,optimizer,print_every):
        self.train()
        t0 = time.time()
        batches = 0
        running_loss = 0
        for inputs, labels in trainloader:
            batches += 1
            #t1 = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            #print('training this batch took {:.3f} seconds'.format(time.time() - t1))
            running_loss += loss
            
            if batches % print_every == 0:
                print(f"{time.asctime()}.."
                        f"Time Elapsed = {time.time()-t0:.3f}.."
                        f"Batch {batches+1}/{len(trainloader)}.. "
                        f"Average Training loss: {running_loss/(batches):.3f}.. "
                        f"Batch Training loss: {loss:.3f}.. "
                        )
                t0 = time.time()
           
        return running_loss/len(trainloader) 

    def validate_(self,validloader):
        running_loss = 0.
        accuracy = 0
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)
        self.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(torch.exp(outputs), 1)
                update_classwise_accuracies(preds,labels,class_correct,class_totals)
            
        accuracy = (100*np.sum(list(class_correct.values()))/np.sum(list(class_totals.values())))    
        self.train()
        return (running_loss/len(validloader),accuracy)
    
    def evaluate(self,testloader):
        self.eval()
        self.model.to(self.device)
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                ps = torch.exp(outputs)
                _, preds = torch.max(ps, 1)
                update_classwise_accuracies(preds,labels,class_correct,class_totals)
                
        self.train()    
        return get_accuracies(self.class_names,class_correct,class_totals)
    
    def predict(self,inputs,topk=1):
        self.eval()
        self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
            ps = torch.exp(outputs)
            p,top = ps.topk(topk, dim=1)
        return p,top
    
    def fit(self,trainloader,validloader,epochs=2,print_every=10,validate_every=1):
        print('fit: best accuracy = {:.3f}'.format(self.best_accuracy))    
        for epoch in range(epochs):
            self.model.to(self.device)
            print('epoch {:3d}/{}'.format(epoch+1,epochs))
            epoch_train_loss =  self.train_(trainloader,self.criterion,
                                            self.optimizer,print_every)
                    
            if  validate_every and (epoch % validate_every == 0):
                t2 = time.time()
                epoch_validation_loss,epoch_accuracy = self.validate_(validloader)
                time_elapsed = time.time() - t2
                print(f"{time.asctime()}--Validation time {time_elapsed:.3f} seconds.."
                      f"Epoch {epoch+1}/{epochs}.. "
                      f"Epoch Training loss: {epoch_train_loss:.3f}.. "
                      f"Epoch validation loss: {epoch_validation_loss:.3f}.. "
                      f"validation accuracy: {epoch_accuracy:.3f}")
                
                if self.best_accuracy == 0. or (epoch_accuracy > self.best_accuracy):
                    print('updating best accuracy: previous best = {:.3f} new best = {:.3f}'.format(self.best_accuracy,
                                                                                     epoch_accuracy))
                    self.best_accuracy = epoch_accuracy
                    torch.save(self.state_dict(),self.best_accuracy_file)
                    
                self.train() # just in case we forgot to put the model back to train mode in validate
                
        print('loading best accuracy model')
        self.load_state_dict(torch.load(self.best_accuracy_file))
                
                
    def set_criterion(self,criterion_name):
            if criterion_name.lower() == 'nllloss':
                self.criterion_name = 'NLLLoss'
                self.criterion = nn.NLLLoss()
            elif criterion_name.lower() == 'crossentropyloss':
                self.criterion_name = 'CrossEntropyLoss'
                self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self,params,optimizer_name='adam',lr=0.003):
        from torch import optim

        if optimizer_name.lower() == 'adam':
            print('setting optim Adam')
            self.optimizer = optim.Adam(params,lr=lr)
            self.optimizer_name = optimizer_name
        elif optimizer_name.lower() == 'sgd':
            print('setting optim SGD')
            self.optimizer = optim.SGD(params,lr=lr)
        elif optimizer_name.lower() == 'adadelta':
            print('setting optim Ada Delta')
            self.optimizer = optim.Adadelta(params)
            
    def set_model_params(self,
                         criterion_name,
                         optimizer_name,
                         lr, # learning rate
                         dropout_p,
                         model_name,
                         best_accuracy,
                         best_accuracy_file,
                         chkpoint_file):
        
        self.criterion_name = criterion_name
        self.set_criterion(criterion_name)
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.parameters(),optimizer_name,lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.best_accuracy = best_accuracy
        #print('set_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))  
        self.best_accuracy_file = best_accuracy_file
        self.chkpoint_file = chkpoint_file
    
    def get_model_params(self):
        params = {}
        params['device'] = self.device
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion_name'] = self.criterion_name
        params['lr'] = self.lr
        params['dropout_p'] = self.dropout_p
        params['best_accuracy'] = self.best_accuracy
        print('get_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))  
        params['best_accuracy_file'] = self.best_accuracy_file
        params['chkpoint_file'] = self.chkpoint_file
        print('get_model_params: chkpoint file = {}'.format(self.chkpoint_file))  
        return params
    
    def save_chkpoint(self):
        saved_model = {}
        saved_model['params'] = self.get_model_params()    
        torch.save(saved_model,self.chkpoint_file)
        print('checkpoint created successfully in {}'.format(self.chkpoint_file))
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class EnsembleModel(Network):
    def __init__(self,models):
        self.criterion = None
        super().__init__()
        self.models = models
        if sum(model[1] for model in models) != 1.0:
            raise ValueError('Weights of Ensemble must sum to 1')
            
        
    def evaluate(self,testloader,metric='accuracy'):
        from collections import defaultdict
        #evaluations = defaultdict(float)
        #num_classes = self.models[0][0].num_outputs
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)

        class_names = self.models[0][0].class_names  
        with torch.no_grad():
            
            for inputs, labels in testloader:
                ps_list = []  
                for model in self.models:
                    model[0].eval()
                    model[0].to(model[0].device)
                    inputs, labels = inputs.to(model[0].device), labels.to(model[0].device)
                    logps = model[0].forward(inputs)
                    ps = torch.exp(logps)
                    ps = ps * model[1] # multiply by model's weight
                    ps_list.append(ps)
                    
                final_ps = ps_list[0]
                for i in range(1,len(ps_list)):
                    final_ps = final_ps + ps_list[i]
                _, final_preds = torch.max(final_ps, 1)
                #print(final_preds)
                update_classwise_accuracies(final_preds,labels,class_correct,class_totals)
        
       
        
        return get_accuracies(class_names,class_correct,class_totals)
                   
    
    def predict(self,inputs,topk=1):
        ps_list = []  
        for model in self.models:
            model[0].eval()
            model[0].to(model[0].device)
            with torch.no_grad():
                inputs = inputs.to(model[0].device)
                outputs = model[0].forward(inputs)
                ps_list.append(torch.exp(outputs)*model[1])
       
        final_ps = ps_list[0]
        for i in range(1,len(ps_list)):
            final_ps = final_ps + ps_list[i]
        
        _,top = final_ps.topk(topk, dim=1)
            
        return top
    
    def forward(self,x):
        outputs = []
        for model in self.models:
             outputs.append(model[0].forward(x))
        return outputs
            
