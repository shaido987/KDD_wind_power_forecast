import os
import torch
import pickle
import numpy as np
import pandas as pd

from methods.mdlinear.exp.exp_main import ExpMain
from methods.xtgn.data_process.data_process import TestData as TestDataWP
from methods.xtgn.model.engine import trainer


def forecast_mdlinear(settings):
    predictions = []
    for horizon in settings['horizons']:
        settings['pred_len'] = horizon

        exp = ExpMain(settings)
        prediction = np.array(exp.predict(settings['model_id'], settings))
        prediction = prediction.reshape(134, -1)[:, -settings['step_size']:]
        predictions.append(prediction)

        torch.cuda.empty_cache()

    predictions = np.expand_dims(np.concatenate(predictions, axis=1), axis=-1)
    return predictions


def forecast_xtgn(args):
    model_dir = os.path.join(args['checkpoints'], args['model_name'])
    model_path = os.path.join(model_dir, 'checkpoint.pth')

    # Load scaler
    ss_path = os.path.join(model_dir, 'scaler.pickle')
    with open(ss_path, 'rb') as handle:
        scaler = pickle.load(handle)

    adj_matrix = pd.read_csv(os.path.join(model_dir, 'adj_matrix.csv'), header=None).to_numpy()
    adj_matrix = torch.Tensor(adj_matrix).to(args['device'])

    # Predict batch size is always 1
    args['batch_size'] = 1

    engine = trainer(device=args['device'], scaler=scaler, num_nodes=args['num_nodes'],
                     seq_length_x=args['seq_length_x'], in_dim=args['feature_dim'], out_dim=args['seq_length_y'],
                     seq_length_y=args['seq_length_y'], weight_decay=args['weight_decay'],
                     dropout_rate=args['dropout_rate'], milestones=args['milestone'], num_epochs=args['max_epoch'],
                     print_freq=args['print_freq'], batch_size=args['batch_size'], gamma=None, clip=None,
                     residual_channels=args['residual_channels'], dilation_channels=args['dilation_channels'],
                     skip_channels=args['skip_channels'], end_channels=args['end_channels'], blocks=args['blocks'],
                     layers=args['wavenet_layers'], kernel_size=args['kernel_size'],
                     learning_rate=args['learning_rate'], embed_dim=args['embed_dim'], adj_matrix=adj_matrix)

    engine.model.load_state_dict(torch.load(model_path))

    seqs, _, _ = TestDataWP(args['path_to_test_x']).get_all_turbines()
    seqs = np.array(seqs)[:, -args['seq_length_y']:, :]

    feature_dims = seqs.shape[-1]
    seqs = scaler.transform(seqs.reshape(-1, feature_dims)).reshape(seqs.shape)

    with torch.no_grad():
        test_x = torch.Tensor(seqs).to(args['device'])[None, :]  # batch size of 1
        test_x = test_x.transpose(1, 3).transpose(2, 3)
        predictions = engine.test(test_x).squeeze().cpu().numpy()

    predictions = np.expand_dims(predictions, axis=-1)
    predictions[predictions < 0] = 0
    predictions[np.isnan(predictions)] = 0
    return predictions


def forecast(settings):
    mdlinear_settings = {**settings['mdlinear'], **settings}
    mdlinear_preds = forecast_mdlinear(mdlinear_settings)

    xtgn_settings = {**settings['xtgn'], **settings}
    xtgn_preds = forecast_xtgn(xtgn_settings)

    predictions = (mdlinear_preds + xtgn_preds) / 2
    return predictions
