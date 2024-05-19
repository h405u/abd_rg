# abd_rg

This is a toy model that **abd**uctively learns **r**egular **g**rammars from raw data.

## Environment

This model has been tested under macOS 14.3 with the following specifications.

- python        3.12.2
- pytorch       2.3.0
- torchvision   0.18.0
- tqdm          4.66.4
- matplotlib    3.8.4
- scikit-learn  1.4.2
- numpy         1.26.4

## Run it yourself

Choose `DEVICE` in `run_abduce.py` or `run_select.py` according to your hardware('mps' for apple chips, 'cuda' if available, 'cpu' for rest circumstances).

```shell
python run_abduce.py
python run_select.py
```
