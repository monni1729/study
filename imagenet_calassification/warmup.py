import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import warnings

####
# 주의 해야할점은 epoch가 50인데 warm-up이 10이면 base_scheduler가 도는건 40뿐임...
# 이걸 어찌 컨트롤할지는... 애초에 warmup-epoch를 max_epoch에 더해야하나...쩝
####

AVAIABLE_SCHEDULER = [
    'constant', 'single_step', 'multi_step', 'cosine', 'constant', 'poly',
    'exponential'
]


class LinearWarmUpLR(_LRScheduler):
    # https://github.com/meetshah1995/pytorch-semseg/blob/89f4abe180528a69e32ac1217746f68dfafd0e36/ptsemseg/schedulers/schedulers.py
    def __init__(self,
                 optimizer,
                 warmup_epoch,
                 base_scheduler=None,
                 gamma=0.1,
                 last_epoch=-1):

        self.warmup_epoch = warmup_epoch
        self.gamma = gamma
        self.verbose = False

        if base_scheduler is None:
            raise ValueError("base scheduler should be defined!!")
        self.scheduler = base_scheduler
        #self.scheduler.last_epoch += self.warmup_epoch # base_scheduler에 warmupepoch를 적용
        self.cold_lrs = self.scheduler.get_last_lr()

        super(LinearWarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epoch:
            alpha = self.last_epoch / float(self.warmup_epoch - 1)
            factor = self.gamma * (1 - alpha) + alpha
            lr_ = [factor * base_lr for base_lr in self.cold_lrs]
            # print(f"warm-up... epoch {self.last_epoch+1}\tlr : {lr_[0]:.5f}")
            return lr_
        else:
            self.scheduler.step()
        return self.scheduler.get_last_lr()


# class WarmUpLR(_LRScheduler):
#     #https://github.com/meetshah1995/pytorch-semseg/blob/89f4abe180528a69e32ac1217746f68dfafd0e36/ptsemseg/schedulers/schedulers.py
#     def __init__(self,
#                  optimizer,
#                  multiplier,
#                  warmup_epoch,
#                  gamma=0.01,
#                  after_scheduler=None,
#                  last_epoch=-1):
#         self.multiplier = multiplier
#         self.warmup_epoch = warmup_epoch
#         self.after_scheduler = after_scheduler

#         self.finished = False
#         self.gamma = gamma
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         print(self.base_lrs)
#         if self.last_epoch > self.warmup_epoch:
#             if self.after_scheduler:
#                 if not self.finished:
#                     self.after_scheduler.base_lrs = [
#                         base_lr * self.multiplier for base_lr in self.base_lrs
#                     ]
#                     self.finished = True
#                     print('end warmup')
#                 return self.after_scheduler.get_last_lr()
#             return [base_lr * self.multiplier for base_lr in self.base_lrs]

#         # return [
#         #     base_lr *
#         #     ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
#         #     for base_lr in self.base_lrs
#         # ]
#         ##############################################
#         ''' '''
#         cold_lrs = self.after_scheduler.get_last_lr()
#         alpha = self.last_epoch / float(self.warmup_epoch)
#         factor = self.gamma * (1 - alpha) + alpha
#         return [factor * base_lr for base_lr in cold_lrs]

#     def step(self, epoch=None, metrics=None):
#         if self.finished and self.after_scheduler:
#             if epoch is None:
#                 self.after_scheduler.step(None)
#             else:
#                 self.after_scheduler.step(epoch - self.warmup_epoch)
#         else:
#             return super(WarmUpLR, self).step(epoch)


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.verbose = False

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 max_iter,
                 decay_iter=1,
                 gamma=0.9,
                 last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        self.verbose = False

        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter))**self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


def build_lr_scheduler(optimizer, scheduler, **kwargs):
    if scheduler not in AVAIABLE_SCHEDULER:
        raise ValueError(
            f'Unsupported scheduler: {scheduler}. Must be one of {AVAIABLE_SCHEDULER}'
        )
    if scheduler == 'constant':
        print(' >>> constant lr scheduler')
        lr_scheduler = ConstantLR(optimizer)
        print(
            f" >>> lr_scheduler is '{scheduler}' with constant lr {optimizer.param_groups[0]['lr']}"
        )

    elif scheduler == 'single_step':
        print(' >>> single step lr scheduler')
        '''
        single step scheduler
        step_size (essential, int)
        gamma (optinal, float, default = 0.1)
        '''
        if 'step_size' not in kwargs:
            raise ValueError("'step_size' is not defined.")
        if not isinstance(kwargs['step_size'], int):
            raise TypeError(
                f"For single_step lr_scheduler, stepsize must be an integer, but got {type(kwargs['step_size'])}"
            )
        step_size = kwargs['step_size']

        try:
            gamma = kwargs['gamma']
        except:
            gamma = 0.1
            print("gamma is not defined. so, gamma is set default value 0.1")

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=step_size,
                                                 gamma=gamma)
        print(
            f" >>> lr_scheduler is '{scheduler}' with single step size {step_size} & gamma {gamma}"
        )

    elif scheduler == 'multi_step':
        print(' >>> multi step lr scheduler')
        '''
        multi step scheduler
        step_size (essential, list[int, ...])
        gamma (optinal, float, default = 0.1)
        '''
        if 'step_size' not in kwargs:
            raise ValueError("'step_size' is not defined.")
        if not isinstance(kwargs['step_size'], list):
            raise TypeError(
                f" >>> For multi_step lr_scheduler, stepsize must be a list, but got {type(kwargs['step_size'])}"
            )
        step_size = kwargs['step_size']

        try:
            gamma = kwargs['gamma']
        except:
            gamma = 0.1
            print(
                " >>> gamma is not defined. so, gamma is set default value 0.1"
            )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=step_size,
                                                      gamma=gamma)

        print(f" >>> lr_scheduler is '{scheduler}' with step size [",
              *step_size, f'] & gamma {gamma}')

    elif scheduler == 'cosine':
        print(' >>> cosine lr scheduler')
        '''
        cosine scheduler 
        max_epoch에 맞춰서 시작부터 last epoch까지 lr을 base_lr~0.0으로 맞춰주는 스케쥴러
        max_epoch (essential, int)
        '''
        if 'max_epoch' not in kwargs:
            raise ValueError(" >>> 'max_epoch' is not defined.")

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(kwargs['max_epoch']))

        print(
            f" >>> lr_scheduler is '{scheduler}' with max_epoch {kwargs['max_epoch']}"
        )

    elif scheduler == 'exponential':
        print(' >>> exponential lr scheduler')
        '''
        exponential scheduler 
        gamma (optional, float, default=0.99)
        '''
        try:
            gamma = kwargs['gamma']
        except:
            gamma = 0.99
            print(
                f" >>> gamma is not defined. so, gamma is set default value {gamma}"
            )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        print(f" >>> lr_scheduler is '{scheduler}' with gamma {gamma}")

    elif scheduler == 'poly':
        print(' >>> poly lr scheduler')
        '''
        poly scheduler, deeplab v3 논문 참고
        iter_per_epoch (essential, int)
        max_epoch (essential, int)
        gamma (optional, float, default=0.9)
        decay_iter (optinal, int, default=1), 1이면 decay iter없음 100이면 100 iter동안 안줄어듦
        '''
        max_iter = kwargs['iter_per_epoch'] * kwargs['max_epoch']
        try:
            gamma = kwargs['gamma']
        except:
            gamma = 0.9
            print(
                f" >>> gamma is not defined. so, gamma is set default value {gamma}"
            )
        try:
            decay_iter = kwargs['decay_iter']
        except:
            decay_iter = 1
            print(' >>> decay_iter in not defined, so don`t do decay')

        lr_scheduler = PolynomialLR(optimizer, max_iter, decay_iter, gamma)
        print(
            f" >>> lr_scheduler is '{scheduler}' with max_iter {max_iter} & gamma {gamma} & decay_iter {decay_iter}"
        )

    if 'warmup' in kwargs and kwargs['warmup']:
        if 'warmup_epoch' not in kwargs:
            raise ValueError(f" >>> 'warmup_epoch' should be defined!!!")

        print(f" >>> Using linear-warmup with {kwargs['warmup_epoch']} epochs")
        lr_scheduler = LinearWarmUpLR(optimizer,
                                      warmup_epoch=kwargs['warmup_epoch'],
                                      base_scheduler=lr_scheduler)
    return lr_scheduler


