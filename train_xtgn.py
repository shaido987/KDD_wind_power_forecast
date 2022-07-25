import os
import time
import pickle
import random
import torch
import numpy as np
import pandas as pd

from methods.xtgn.data_process.metrics import metric
from methods.xtgn.data_process.data_process import run_data_preprocess
from methods.xtgn.data_process.adj_calculation import compute_adj_matrix
from methods.xtgn.data_process.util import EarlyStopping, load_wp_dataset_mask
from methods.xtgn.model.engine import trainer
from methods.prepare import prep_env


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args, model_dir):
    model_path = os.path.join(model_dir, 'checkpoint.pth')

    # Load scaler
    ss_path = os.path.join(model_dir, 'scaler.pickle')
    with open(ss_path, 'rb') as handle:
        scaler = pickle.load(handle)

    device = args['device']
    data_loader, _ = load_wp_dataset_mask(model_dir, args['ratio'], args['batch_size'], args['batch_size'],
                                          scaler=scaler)
    print('The Total Parameter: \n', args)

    adj_path = os.path.join(model_dir, 'adj_matrix.csv')
    adj_matrix = pd.read_csv(adj_path, header=None).to_numpy()
    adj_matrix = torch.Tensor(adj_matrix).to(device)

    engine = trainer(device=args['device'], scaler=scaler, num_nodes=args['num_nodes'],
                     seq_length_x=args['seq_length_x'], in_dim=args['feature_dim'], out_dim=args['seq_length_y'],
                     seq_length_y=args['seq_length_y'], weight_decay=args['weight_decay'],
                     dropout_rate=args['dropout_rate'], milestones=args['milestone'], num_epochs=args['max_epoch'],
                     print_freq=args['print_freq'], batch_size=args['batch_size'], gamma=None, clip=None,
                     residual_channels=args['residual_channels'], dilation_channels=args['dilation_channels'],
                     skip_channels=args['skip_channels'], end_channels=args['end_channels'], blocks=args['blocks'],
                     layers=args['wavenet_layers'], kernel_size=args['kernel_size'],
                     learning_rate=args['learning_rate'], embed_dim=args['embed_dim'], adj_matrix=adj_matrix)

    print('Start Training: ', flush=True)
    his_loss = []

    train_time, val_time = [], []
    early_stopping = EarlyStopping(patience=1, verbose=True, model_save_path=model_path)

    for i in range(1, args['max_epoch'] + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        data_loader['train_loader'].shuffle()

        # Train model
        for iter, (x, y, m) in enumerate(data_loader['train_loader'].get_iterator()):

            train_x = torch.Tensor(x).to(device).transpose(1, 3)
            train_y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            train_m = torch.Tensor(m).to(device).transpose(1, 3)[:, 0, :, :]
            metrics = engine.train(train_x, train_y, train_m, ite=i)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args['print_freq'] == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # Validation
        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for iter, (x, y, m) in enumerate(data_loader['test_loader'].get_iterator()):
            valid_x = torch.Tensor(x).to(device).transpose(1, 3)
            valid_y = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            valid_m = torch.Tensor(m).to(device).transpose(1, 3)[:, 0, :, :]
            metrics = engine.eval(valid_x, valid_y, valid_m, ite=i)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mean_train_loss = np.mean(train_loss)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)
        his_loss.append(mean_valid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
              'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mean_train_loss, mean_train_mape, mean_train_rmse, mean_valid_loss, mean_valid_mape,
                         mean_valid_rmse, (t2 - t1)), flush=True)

        early_stopping(mean_valid_loss, engine.model)
        if early_stopping.early_stop:
            print('Early Stopping!')
            break

    print("Training finished")
    print("The valid loss on best model is: ", early_stopping.val_loss_min)

    # Testing
    engine.model.load_state_dict(torch.load(model_path))

    outputs = []
    real_y = torch.Tensor(data_loader['y_test']).to(device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]
    real_m = torch.Tensor(data_loader['m_test']).to(device)
    real_m = real_m.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, m) in enumerate(data_loader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(device)
        test_x = test_x.transpose(1, 3)
        test_m = torch.Tensor(m).to(device).transpose(1, 3)[:, 0, :, :]
        with torch.no_grad():
            preds = engine.test(test_x, test_m)
        outputs.append(preds.squeeze())

    y_hat = torch.cat(outputs, dim=0)
    y_hat = y_hat[:real_y.size(0), ...]

    amae, amape, armse = [], [], []
    for i in range(args['seq_length_y']):
        pred = y_hat[:, :, i]
        real = real_y[:, :, i]
        m = real_m[:, :, i]
        metrics = metric(pred, real, m, 0.0)
        log = 'Evaluate best model on test data for horizon {:d}, ' \
              'Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0].item())
        amape.append(metrics[1].item())
        armse.append(metrics[2].item())

    log = 'On average over 288 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print('The best model had been saved! you can ues it to inference your data.')

    metrics = metric(y_hat / 1000, real_y / 1000, real_m,  0.0)
    mae = metrics[0].cpu().numpy()
    rmse = metrics[2].cpu().numpy()
    score = (mae + rmse) / 2
    print('The model result of MAE:{}  RMSE:{}  Score:{}'.format(mae, rmse, score))


if __name__ == '__main__':
    args = prep_env()
    args = {**args['xtgn'], **args}
    model_dir = os.path.join('methods', args['checkpoints'], args['model_name'])
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Check if the data has been preprocessed or not.
    adj_matrix_path = os.path.join(model_dir, 'adj_matrix.csv')
    if not os.path.isfile(adj_matrix_path):
        input_data = os.path.join(args['data_path'], args['filename'])
        location_data = os.path.join(args['data_path'], args['location_file'])
        run_data_preprocess(input_data, model_dir, lag=args['seq_length_x'], horizon=args['seq_length_y'],
                            ratio=args['ratio'])
        compute_adj_matrix(location_data, adj_matrix_path)

    main(args, model_dir)
