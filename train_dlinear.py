import random
import torch
import numpy as np

from dlin.exp.exp_main import Exp_Main
from prepare import prep_env


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = prep_env()
    args = {**args['dlin'], **args}

    # setting record of experiments
    for horizon in args['horizons']:

        args["pred_len"] = horizon
        args["output_len"] = horizon

        exp = Exp_Main(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args['model_id']))
        exp.train(args['model_id'])
        torch.cuda.empty_cache()
