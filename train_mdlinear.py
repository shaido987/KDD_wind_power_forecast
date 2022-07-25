import os
import torch
import random
import numpy as np

from methods.mdlinear.exp.exp_main import ExpMain
from methods.prepare import prep_env


if __name__ == '__main__':
    args = prep_env()
    args = {**args['mdlinear'], **args}
    args['checkpoints'] = os.path.join('methods', args['checkpoints'])

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    for horizon in args['horizons']:
        args["pred_len"] = horizon

        exp = ExpMain(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args['model_id']))
        exp.train(args['model_id'])
        torch.cuda.empty_cache()
