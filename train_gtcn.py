import os
import time
import pickle
import random
import torch
import numpy as np

from gtcn.data_process.util import load_wp_dataset
from gtcn.data_process.early_stopping import EarlyStopping
from gtcn.data_process.data_process import preprocess_data
from gtcn.model.engine import trainer
from gtcn.data_process.metrics import metric
from prepare import prep_env


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = prep_env()
    args = {**args['gtcn'], **args}

    setup_seed(seed=args['seed'])
    print('Using random seed is: {}'.format(args['seed']))

    data_file = os.path.join(args['data_path'], args['filename'])
    data = preprocess_data(data_file, args['npz_path'], args['seq_length_y'], args['seq_length_x'])
    data_loader, _ = load_wp_dataset(data, args['batch_size'], args['batch_size'], args['batch_size'])
    scaler = data_loader['scaler']
    print('The Total Parameter: \n', args)

    model_dir = os.path.join(args['checkpoints'], args['model_name'])
    model_path = os.path.join(model_dir, 'checkpoint.pth')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("Saving scaler")
    scaler_path = os.path.join(model_dir, 'scaler.pickle')
    with open(scaler_path, 'wb') as handle:
        pickle.dump(scaler, handle, protocol=4)

    adj_matrix = None
    engine = trainer(device=args['device'], scaler=scaler, num_nodes=args['num_nodes'], seq_length_x=args['seq_length_x'],
                     in_dim=args['feature_dim'], out_dim=args['seq_length_y'], seq_length_y=args['seq_length_y'],
                     weight_decay=args['weight_decay'], dropout_rate=args['dropout_rate'], milestones=args['milestone'],
                     gamma=args['gamma'], num_epochs=args['max_epoch'], print_freq=args['print_freq'], clip=args['clip'],
                     batch_size=args['batch_size'], residual_channels=args['residual_channels'],
                     dilation_channels=args['dilation_channels'], skip_channels=args['skip_channels'],
                     end_channels=args['end_channels'], blocks=args['blocks'], layers=args['layers'],
                     kernel_size=args['kernel_size'], learning_rate=args['learning_rate'], embed_dim=args['embed_dim'],
                     adj_matrix=adj_matrix)

    print('Start Training: ', flush=True)
    his_loss = []

    early_stopping = EarlyStopping(patience=5, verbose=True, model_save_path=model_path)

    train_time = []
    val_time = []
    val_loss_min = np.inf
    device = args['device']
    for i in range(1, args['max_epoch'] + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        data_loader['train_loader'].shuffle()

        ## train model
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

        # engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, m) in enumerate(data_loader['val_loader'].get_iterator()):
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

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    engine.model.load_state_dict(torch.load(model_path))

    outputs = []
    real_y = torch.Tensor(data_loader['y_test']).to(device)
    if real_y.shape[0] == 0:
        print('No test data.')
        return
    real_y = real_y.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, m) in enumerate(data_loader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(device)
        test_x = test_x.transpose(1, 3)
        with torch.no_grad():
            preds = engine.test(test_x)
        outputs.append(preds.squeeze())

    y_hat = torch.cat(outputs, dim=0)
    y_hat = y_hat[:real_y.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is: ", val_loss_min)

    amae = []
    amape = []
    armse = []
    for i in range(args['seq_length_y']):
        # pred = scaler.inverse_transform(y_hat[:, :, i])
        pred = y_hat[:, :, i]
        real = real_y[:, :, i]
        metrics = metric(pred, real, 0.0)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0].item())
        amape.append(metrics[1].item())
        armse.append(metrics[2].item())

    log = "On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}"
    print(log.format(args['seq_length_y'], np.mean(amae), np.mean(amape), np.mean(armse)))
    print('The best model had been saved! you can ues it to inference your data.')

    metrics = metric(y_hat / 1000, real_y / 1000, 0.0)
    print('The model result of RMSE:{}'.format(metrics[2].cpu().numpy()))


if __name__ == '__main__':
    print("Running model training")
    begin_time = time.time()
    main()
    end_time = time.time()
    print('Total time spent: {:.4f}'.format(end_time - begin_time))
