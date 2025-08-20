# AlphaEval

The implementation of [AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining](https://arxiv.org/abs/2508.13174).

## Overview

This repository contains implementations of various factor mining models and the AlphaEval evaluation framework. The codebase is organized into two main components:

1. **Factor Mining Models**: Algorithms for discovering trading factors.  
2. **AlphaEval Evaluation Model**: A backtesting and evaluation framework to assess the performance of generated factors.

## Acknowledgements

This project reuses ideas and code from the following open-source projects, to whose authors we extend our sincere thanks:

- **gplearn**  
- **AlphaGen**  
- **AlphaForge**  
- **AlphaQCM**  

## Data Preparation

In a manner similar to [AlphaGen](https://github.com/RL-MLDM/alphagen), we leverage [Qlib](https://github.com/microsoft/qlib#data-preparation) for data storage. and pull our data from the free, open-source [BaoStock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5) service. After installing Qlib and baostock, run the script `data_collection/fetch_baostock_data.py` to download the data. If it is invalid, there is also other data preparation method on the website [Qlib](https://github.com/microsoft/qlib#data-preparation)

The next, Modify the correspoding `path/to/your/qlib_data` in all python files to the data you downloaded.


## Factor Mining Models

The following factor mining models have been implemented or reproduced by the authors of this project:

- **gplearn** (including Random Baseline)  
- **AutoAlpha**  
- **AlphaEvolve**  
- **Fama**  
- **AlphaAgent**  

> **Running Instructions** for the above models:

```bash
python gplearn.py --start_time 2010-01-01 --end_time 2019-12-31 --population_size 1000 --hall_of_fame 50 --n_components 10 --generations 5
python autoalpha.py --start_time 2010-01-01 --end_time 2019-12-31 --population_size 1000 --hall_of_fame 50 --n_components 10 --generations 5
python alphaevolve.py --start_time 2010-01-01 --end_time 2019-12-31 --population_size 1000 --hall_of_fame 50 --n_components 10 --generations 5
python fama.py
python alphaagent.py
```

The code for the following open-source projects is used directly from their original repositories. For setup and usage instructions, please refer to the README files in their respective folders. Copyright remains with the original authors:

- **AlphaGen**  
- **AlphaForge**  
- **AlphaQCM**  


## AlphaEval Evaluation Model

Once you have generated a set of candidate factors, you can evaluate their performance using the AlphaEval framework located in `backtest/modeltester`. A simplified working example is provided in the Jupyter notebook:

```text
backtest/test.ipynb
```
**Special Note:** For the AlphaEvolve project, we have created a custom `my_qlib` to support new operators such as “RelationRank” incorporating the additional operators introduced in the AlphaEvolve paper. During testing, please use `my_modeltester` alongside it.
