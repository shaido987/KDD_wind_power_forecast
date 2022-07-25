# KDD Cup 2022 Wind Power Forecast

This is our solution to the KDD Cup 2022 spatial dynamic wind power forecast challenge.  
Team name: didadida_hualahuala  
Placement: 7th (of 2490 teams)  
Final score (3rd phase): -45.18139

The solution uses a combination of two models: MDLinear and GTCN, see the technical report [to be added].
The trained models used for the final score are included in this repository.

## Training
The training data can be downloaded on the competition website: https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets
Put this file into the data folder before starting to train the models.

All parameter settings are adjusted in the [`methods/prepare.py`](methods/prepare.py) file. The default settings were used for the competition results.  
To train the models, run 

```python
python train_mdlinear.py
```
and
```python
python train_xtgn.py
```
in any order. The trained models and any relevant files are saved to the [`methods/checkpoints`](methods/checkpoints) folder (this folder is shared for both methods).

## Forecast 

To evaluate our method, we use the provided test dataset (in [`data/test_x`](data/test_x) and [`data/test_y`](data/test_y)). The input data contains 14 days and since we do not require that much we use a sliding window to create more test data (see the techincal report). The code for this is included in [`data/split_test_file.py`](./data/split_test_file.py). To use the single test file instead, adjust the values of `path_to_test_x` and `path_to_test_y` in [`methods/prepare.py`](methods/prepare.py).

To run the forecast and evaluate the score, use:
```python
python evaluate.py
```

