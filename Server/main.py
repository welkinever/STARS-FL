import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data_har import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation
from metrics import Metric
from utils_new import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
import csv
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--algo', default='order-layerwise', type=str)
parser.add_argument('--lwr', default='1_1_1_1', type=str)
parser.add_argument('--thres', default=0.00001, type=float)

args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}
cfg['K']=10
cfg['algo'] = args['algo']
cfg['layerwise_ratio'] = [float(i) for i in args['lwr'].split('_')]
THRES = args['thres']

if cfg['model_name'] == 'conv':
    param_num_in_MB = 5.939
elif cfg['model_name'] == 'resnet18':
    param_num_in_MB = 42.618
elif cfg['model_name'] == 'conv_har':
    param_num_in_MB = 17.318
else:
    pass

device_model_time={}

with open("./device_list/STARS/"+cfg['model_name']+"_device_model_time.csv","r") as f:
    reader=csv.reader(f)
    reader_row=next(reader)
    for row in reader:
        device_model_time[str(row[0])+str(float(row[1]))]=float(row[2])*int(cfg['K'])

device_list_device=[]
device_list_modelsize=[]
device_list_commrate=[]
device_list_commtime=[]

with open("./device_list/STARS/"+cfg['model_name']+"_device_list.csv","r") as f:
    reader=csv.reader(f)
    reader_row=next(reader)
    for row in reader:
        device_list_device.append(row[1])
        device_list_modelsize.append(float(row[2]))
        device_list_commrate.append(float(row[3]))
        device_list_commtime.append(param_num_in_MB / (float(row[3])/8/1000/1000))


print('device_list_commtime', device_list_commtime)

cfg['id']=args['id']
print('device_model_time', device_model_time)

max_time_per_round = np.zeros(20)
for n in range(20):
    max_time_per_round[n]=device_model_time[device_list_device[n]+"1.0"]+device_list_commtime[n]
max_round_time = max(max_time_per_round)

def main():
    device = torch.cuda.current_device()
    print(device)
    process_control()
    print(cfg)
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return
        
def quantization(W,N):
    ans=np.zeros(N)
    for n in range(N):
        if W[n]>=0.8:
            ans[n]=1
        if W[n]<0.8 and W[n]>=0.6:
            ans[n]=0.5
        if W[n]<0.6 and W[n]>=0.4:
            ans[n]=0.25
        if W[n]<0.4 and W[n]>=0.2:
            ans[n]=0.125
        if W[n]<0.2:
            ans[n]=0.0625
    return ans
    
def update_model_rate(model_rate, fisher_record, cfg, device_model_time, device_list_device,device_list_commtime):
    last_model_rate=model_rate
    last_T=np.ones(cfg['num_users'])
    for n in range(0,cfg['num_users']):
        if abs(last_model_rate[n]-0.0625)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.0625"]+device_list_commtime[n]*(last_model_rate[n])**2
        if abs(last_model_rate[n]-0.125)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.125"]+device_list_commtime[n]*(last_model_rate[n])**2
        if abs(last_model_rate[n]-0.25)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.25"]+device_list_commtime[n]*(last_model_rate[n])**2
        if abs(last_model_rate[n]-0.5)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.5"]+device_list_commtime[n]*(last_model_rate[n])**2
        if abs(last_model_rate[n]-1)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"1.0"]+device_list_commtime[n]*(last_model_rate[n])**2
       
    print('last_T: ', last_T)  
    W=copy.deepcopy(last_model_rate)
                
    for n in range(0,cfg['num_users']):
        if len(fisher_record) == 0:
            return last_model_rate
        else:          
            if np.mean(fisher_record[-1]) >= THRES:
                W[n] += W[n]*(np.mean(max_time_per_round)/max_time_per_round[n])
            else:
                W[n] -= (W[n]/2)*(max_time_per_round[n]/np.mean(max_time_per_round))

    model_rate=quantization(W,cfg['num_users'])
    return model_rate


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()

    data_split_temp=data_split["train"]
    data_split_len=[0 for n in range(0,cfg['num_users'])]
    for i in range(0,cfg['num_users']):
        data_split_len[i]=len(data_split_temp[i])
    
    maxxx = 0
    MAX_TIME=[]
    Accuracy=[]
    LOSS1 = np.ones(cfg['num_users']) # 
    LOSS2 = np.ones(cfg['num_users']) # 
    LOSS_util = np.ones(cfg['num_users'])
    LOSS_record = []
    fisher_record = []
    fisher = np.zeros(cfg['num_users']) 
    delta_loss_record = []
    delta_loss_record_sum = 0
    W =  np.zeros(cfg['num_users'])

    T =  np.zeros(cfg['num_users'])
    T_comp = np.zeros(cfg['num_users'])
    T_comm = np.zeros(cfg['num_users'])
    modelrate=[0.0625 for n in range(cfg['num_users'])]
    Modelsize_record = np.zeros(cfg['num_users'])
    os.makedirs('./log/'+cfg['id'], exist_ok=True)
    for n in range(cfg['num_users']):
        T[n]=device_model_time[device_list_device[n]+"1.0"]+device_list_commtime[n]
        T_comp[n]=device_model_time[device_list_device[n]+"1.0"]
        T_comm[n]=device_list_commtime[n]
    for epoch in range(1, 1000):
        logger.safe(True)
        modelrate=update_model_rate(modelrate, fisher_record,cfg,device_model_time, device_list_device,device_list_commtime)

        for n in range(cfg['num_users']):
            T[n]=device_model_time[device_list_device[n]+str(modelrate[n])]+device_list_commtime[n]*(modelrate[n])**2
            T_comp[n]=device_model_time[device_list_device[n]+str(modelrate[n])]
            T_comm[n]=device_list_commtime[n]*(modelrate[n])**2
            Modelsize_record[n]=device_list_modelsize[n]*(modelrate[n])**2

        federation = Federation(global_parameters, modelrate, label_split)
        LOSS1 = LOSS2.copy()
        LOSS2, maxxx, fisher, LOSS_util = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS2, T, cfg['K'],fisher,cfg,LOSS_util)
        fisher_record.append(fisher)
        LOSS_record.append(LOSS_util)
        delta_loss_record.append(np.mean(abs(LOSS2-LOSS1)))
        delta_loss_record_sum+=np.mean(abs(LOSS2-LOSS1))
        test_model = stats(dataset['train'], model)
        thisacc=test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch, Accuracy)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS, T, K,fisher, cfg, LOSS_util):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, K)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    max_time = 0
    for m in range(num_active_users):
        local_parameters[m],LOSS[user_idx[m]],fisher[user_idx[m]], LOSS_util[user_idx[m]] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger, cfg))
        local_time = T[m]
        if local_time > max_time:
            max_time = local_time
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m+1, num_active_users),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Training Time: {}'.format(local_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])
    maxxx = maxxx + max_time
    MAX_TIME.append(maxxx)
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    return LOSS, maxxx, fisher, LOSS_util


def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            if 'har' not in cfg['data_name']:
                input = collate(input)
            else:
                input = {'img': input[0], 'label': input[1]}
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch, Accuracy):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        data_loader = make_data_loader({'test': dataset})['test']
        Sum = 0
        for i, input in enumerate(data_loader):
            if 'har' not in cfg['data_name']:
                input = collate(input)
            else:
                input = {'img': input[0], 'label': input[1]}
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            Sum = Sum + evaluation['Global-Accuracy']
            logger.append(evaluation, 'test', input_size)
        Accuracy.append(Sum/(i+1))
        thisacc=Accuracy[-1]
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return thisacc


def make_local(dataset, data_split, label_split, federation, K):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]], K)
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split, K):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.K =K

    def train(self, local_parameters, lr, logger, cfg):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        Loss = 0
        Loss_util = 0
        optimizer = make_optimizer(model, lr)
        cnt = 0
        fisher = []

        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                cnt += 1
                if 'har' not in cfg['data_name']:
                    input = collate(input)
                else:
                    input = {'img': input[0], 'label': input[1]}
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_square_mean = torch.mean(param.grad ** 2).item() / input_size
                        fisher.append(float(grad_square_mean))
                

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
                Loss += evaluation['Local-Loss']
                Loss_util += evaluation['Local-Loss']**2
                if cnt >= self.K:
                    break
            if cnt >= self.K:
                break

        Loss = Loss / self.K
        Loss_util=Loss_util/(cfg['batch_size']['train']*self.K)
        local_parameters = model.state_dict()
        max_fisher=max(fisher)
        return local_parameters, Loss, max_fisher, Loss_util



if __name__ == "__main__":
    main()
