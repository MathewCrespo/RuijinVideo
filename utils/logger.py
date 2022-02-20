import os,sys
sys.path.append('../')
from shutil import copy, rmtree
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utility import *

class Logger(object):
    """
    My specific logger.
    Args:
        logdir: (str)
    """
    def __init__(self, logdir, ts_dir="tensorboard"):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.logfile = open(os.path.join(logdir, 'log.txt'), 'a+')
        self.summary_writer = SummaryWriter(os.path.join(logdir, ts_dir))
        self.global_step = 0
        self.global_iter = 0
        self.inner_iter = 0

    def log_string(self, out_str):
        self.logfile.write(str(out_str) + '\n')
        self.logfile.flush()
        print(out_str)

    def log_dict(self, args, prefix=''):
        """
        Recursively print and log the configs in a dict.
        Args:
            args: (dict)
            prefix: (str)
        """
        for k,v in args.items():
            if isinstance(v,dict):
                self.log_dict(v, prefix + k + '.')
            else:
                self.logfile.write(prefix + '{:30} {}\n'.format(k,v))
                print(prefix + '{:30} {}\n'.format(k,v))

    def log_config(self, config):
        """
        print and log configs. If configs is an object,
        must provide __dict__ property.
        """
        if isinstance(config, dict):
            self.log_dict(config)
        else:
            self.log_dict(config.__dict__)

    def backup_files(self, file_list):
        for filepath in file_list:
            copy(filepath, self.logdir)

    def auto_backup(self, root='./'):
        for f_name in os.listdir(root):
            if f_name.endswith('.py'):
                save_path = os.path.join(os.path.join(self.logdir,'src'), root)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                copy(os.path.join(root,f_name), save_path)
            elif os.path.isdir(f_name):
                if not "configs" in f_name: ##only copy the used config
                    self.auto_backup(f_name)

    def close(self):
        self.logfile.close()
        self.summary_writer.close()

    def log_scalar(self, tag, value, global_step=None, print=True):
        if global_step is None:
            self.summary_writer.add_scalar(tag, value, self.global_step)
            if print:
                self.log_string("{}: {}\n".format(tag, value))
        else:
            self.summary_writer.add_scalar(tag, value, global_step)
            if print:
                self.log_string("{}:{}\n".format(tag, value))
            
        

    def log_scalar_train(self, tag, value, global_step):
        self.summary_writer.add_scalar('train/'+tag, value, global_step)
        self.summary_writer.flush()

    def log_histogram_train(self, tag, value, global_step):
        self.summary_writer.add_histogram('train/'+tag,value,global_step)

    def log_scalar_eval(self, tag, value, global_step):
        self.summary_writer.add_scalar('eval/'+tag, value, global_step)
        self.summary_writer.flush()

    def save(self, net, optimizer, lrsch, criterion, global_step=None):
        '''
        save the model/optimizer checkpoints with global step
        param net: (nn.Module) network
        param optimizer: (nn.optim.optimizer)
        param lrsch: (nn.optim.lr_scheduler)
        param criterion: (nn.Module) loss function
        '''
        if not isinstance(global_step, int):
            global_step = self.global_step

        self.log_string('Saving{}'.format(global_step))
        state_net = {'net': net.state_dict()}
        state_optim = {
            'opt': optimizer.state_dict(),
            'epoch': global_step,
            'loss': None if criterion is None else criterion.state_dict(),
            'sch': lrsch.state_dict(),
        }
        save_dir = os.path.join(self.logdir, 'ckp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            torch.save(state_net,
                       os.path.join(save_dir, 'net.ckpt{}.pth'.format(global_step)))
            torch.save(state_optim,
                       os.path.join(save_dir, 'optim.ckpt{}.pth'.format(global_step)))
            return True
        except:
            print('save failed!')
            return False
    
    def save_result(self, dir_name="results", data=None):
        save_dir = os.path.join(self.logdir, dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            torch.save(data, os.path.join(save_dir, "res{}.pth".format(self.global_step)))
            return True
        except:
            print("save failed!")
            return False

    def load_result(self, data, dir_name="results", global_step=None, key=None):
        """
        Load an extra data from the resume config. This is a safe loading. 
        If global_step is invalid, it would not change data.

        Args:
            key: (str) if given, load a speical key from result
        """
        if global_step > 0:
            load_dir = os.path.join(self.logdir, dir_name, "res{}.pth".format(global_step))
            if key is not None:
                return torch.load(load_dir)[key]
            return torch.load(load_dir)
        else:
            return data

    def load(self, net, optimizer, lrsch, criterion, global_step=None):
        '''
        Load network and optimizing state given global step.
        '''
        net = self.load_net(net, global_step)
        optimizer, lrsch, criterion = self.load_optim(optimizer, lrsch, criterion, global_step)
        return net, optimizer, lrsch, criterion

    def load_net(self, net, global_step=None):
        """
        Load network. This is a sub-function of self.load()
        """
        if global_step > 0:
            self.global_step = global_step
            load_dir = os.path.join(self.logdir,'ckp')
            ckpt_path = os.path.join(load_dir, 'net.ckpt{}.pth'.format(global_step))
            self.log_string('==> Resuming net of epoch {}'.format(
                ckpt_path
            )
            )
            kwargs = {'map_location': lambda storage, loc: storage}
            ckpt_net = torch.load(
                ckpt_path,
                **kwargs
            )
            ckpt_net['net'] = {k.replace('module.', ''): v for k, v in ckpt_net['net'].items()}
            net.load_state_dict(ckpt_net['net'], strict=False)
            return net
        else:
            print("Warning: not implemented network loading, return default things\n")
            return net
    
    def load_optim(self, optimizer, lrsch, criterion, global_step=None):
        """
        Load optimizing state including optimizer, learning scheduler,
        criterion. This is a sub-function of self.load()
        """
        if global_step > 0:
            self.global_step = global_step
            load_dir = os.path.join(self.logdir,'ckp')
            optim_path = os.path.join(load_dir, 'optim.ckpt{}.pth'.format(global_step))
            self.log_string('==> Resuming Optimim of epoch {}'.format(
                optim_path)
            )
            kwargs = {'map_location': lambda storage, loc: storage}
            ckpt_optim = torch.load(optim_path)
            optimizer.load_state_dict(ckpt_optim['opt'])
            start_epoch = ckpt_optim['epoch']
            lrsch.load_state_dict(ckpt_optim['sch'])
            criterion.load_state_dict(ckpt_optim['loss'])
            return optimizer, lrsch, criterion
        else:
            print("Warning: not implemented optimizing loading, return default things\n")
            return optimizer, lrsch, criterion

    
    def add_graph(self, model):
        try:
            self.summary_writer.add_graph(model, torch.zeros(1,3,32,32).cuda())
        except:
            self.log_string('Failed to add graph to tensorboard.')
        FILENAME = os.path.join(self.logdir, 'model_info.txt')
        self.log_string('Write model architecture to {}.'.format(FILENAME))
        if not os.path.exists(FILENAME):
            with open(FILENAME, 'w') as f:
                f.write('Model Architecture:\n')
                f.write(str(model))
                f.write('\n\nTrainable Parameters:\n')
                for p in model.named_parameters():
                    if p[1].requires_grad:
                        f.write('{} -> {} ({})\n'.format(p[0], p[1].size(), count_params(p[1])))
                f.write('\n\nNumber of trainable parameters: {}'.format(count_params(model)))  
            
    def update_step(self, global_step=None):
        if not isinstance(global_step, int):
            self.global_step += 1
        else:
            self.global_step = global_step

    def update_iter(self):
        self.global_iter += 1
        self.inner_iter += 1
    
    def clear_inner_iter(self):
        self.inner_iter = 0

    def clear_iter(self):
        self.global_iter = 0

