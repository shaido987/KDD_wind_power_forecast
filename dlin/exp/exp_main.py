import os
import time
import pickle
import warnings
import numpy as np
import torch
from torch import optim

from dlin.exp.exp_basic import Exp_Basic
from dlin.data_provider.data_factory import data_provider
from dlin.models import DLinear
from dlin.utils.tools import EarlyStopping, adjust_learning_rate, CustomMaskedLoss

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = DLinear.Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'],
                                 weight_decay=self.args['weight_decay'])
        return model_optim

    def _get_checkpoint_path(self, setting):
        path = os.path.join(self.args['checkpoints'], setting)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _save_scaler(self, path, train_data):
        if train_data.scaler is not None:
            ss_path = os.path.join(path, 'scaler.pickle')
            with open(ss_path, 'wb') as handle:
                pickle.dump(train_data.scaler, handle, protocol=4)
                self.args['scaler'] = train_data.scaler

    def _load_model_and_scaler(self, path, settings):
        name = f"checkpoint_horizon_{self.args['pred_len']}.pth"
        best_model_path = path + '/' + name
        self.model.load_state_dict(torch.load(best_model_path))

        # Load scaler
        if settings['scale']:
            ss_path = os.path.join(path, 'scaler.pickle')
            with open(ss_path, 'rb') as handle:
                self.args['scaler'] = pickle.load(handle)
        else:
            self.args['scaler'] = None

    def validate(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y, mask)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, model_name):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = self._get_checkpoint_path(model_name)
        self._save_scaler(path, train_data)

        name = f"checkpoint_horizon_{self.args['pred_len']}.pth"
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=True)
        model_optim = self._select_optimizer()
        criterion = CustomMaskedLoss(self.args['pred_len'])

        for epoch in range(self.args['train_epochs']):
            train_loss, train_loss_part = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, mask) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y, mask)
                train_loss_part.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 10000 == 0:
                    print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                    print(f"\ttrain loss part: {np.average(train_loss_part)}")
                    train_loss.extend(train_loss_part)
                    train_loss_part = []

            train_loss.extend(train_loss_part)
            train_loss = np.average(train_loss)
            validation_loss = self.validate(vali_loader, criterion)

            print(f"Epoch: {epoch + 1} | Cost time: {time.time() - epoch_time}")
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss} Validation Loss: {validation_loss}")
            early_stopping(validation_loss, self.model, path, name)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + name
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def predict(self, model_name, settings):
        path = self._get_checkpoint_path(model_name)
        self._load_model_and_scaler(path, settings)

        pred_data, pred_loader = self._get_data(flag='pred')
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_x in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()

                if settings['scale']:
                    pred = pred_data.inverse_transform(pred)
                predictions.append(pred[:, :, -1].reshape(-1))  # Only return the target variable

        predictions = np.array(predictions)
        predictions[predictions < 0] = 0
        predictions[np.isnan(predictions)] = 0
        predictions = np.expand_dims(predictions, axis=-1)
        return predictions
