from mylib.model import *
from mylib.fc import *
from mylib.cv_model import *
from mylib.utils import *

def load_chkpoint(chkpoint_file):
        
    restored_data = torch.load(chkpoint_file)

    params = restored_data['params']
    print('load_chkpoint: best accuracy = {:.3f}'.format(params['best_accuracy']))  
    
    if params['model_type'].lower() == 'classifier':
        net = FC( num_inputs=params['num_inputs'],
                  num_outputs=params['num_outputs'],
                  layers=params['layers'],
                  device=params['device'],
                  criterion_name = params['criterion_name'],
                  optimizer_name = params['optimizer_name'],
                  model_name = params['model_name'],
                  lr = params['lr'],
                  dropout_p = params['dropout_p'],
                  best_accuracy = params['best_accuracy'],
                  best_accuracy_file = params['best_accuracy_file'],
                  chkpoint_file = params['chkpoint_file'],
                  class_names =  params['class_names']
          )
    elif params['model_type'].lower() == 'transfer':
        net = TransferNetworkImg(criterion_name = params['criterion_name'],
                                 optimizer_name = params['optimizer_name'],
                                 model_name = params['model_name'],
                                 lr = params['lr'],
                                 device=params['device'],
                                 dropout_p = params['dropout_p'],
                                 best_accuracy = params['best_accuracy'],
                                 best_accuracy_file = params['best_accuracy_file'],
                                 chkpoint_file = params['chkpoint_file'],
                                 head = params['head']
                               )
    
        


    net.load_state_dict(torch.load(params['best_accuracy_file']))

    net.to(params['device'])
    
    return net
    
