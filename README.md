# abd_rg

This is a toy model that **abd**uctively learns **r**egular **g**rammars from raw data.

> The hottest new programming language is English
>
> *Andrej Karpathy*

## Environment

This model has been tested under macOS 14.3 with the following specifications.

- python 3.11.4
- pytorch 2.1.0
- torchvision 0.15.2
- tqdm 4.65.0

## Run it yourself

Choose `DEVICE` in `run_abd_rg.py` according to your hardware('mps' for apple chips, 'cuda' if available, 'cpu' for rest circumstances).

```shell
python run_abd_rg.py
