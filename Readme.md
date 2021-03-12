This repository contains the code to reproduce the results from [Understanding the Extent to which Summarization Evaluation Metrics Measure the Information Quality of Summaries](https://arxiv.org/abs/2010.12495).
The code is based on an early version of the [SacreROUGE](https://github.com/danieldeutsch/sacrerouge) library, which has changed somewhat significantly since.
The version of ROUGE which was decomposed into different parts based on POS, dependency labels, etc., has also been included in SacreROUGE.
See [here](https://github.com/danieldeutsch/sacrerouge/blob/master/doc/metrics/decomposed-rouge.md).

## Environment
First, run these commands to set up the environment:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data
Then prepare the data.
We use the outputs from two models on the CNN/DailyMail dataset, which can be setup using this command:
```
sh datasets/cnndm/setup.sh
```
The data will be downloaded and reformatted.

The other datasets are TAC 2008 and 2009.
Due to license restrictions, we cannot release these datasets.
However, if you have access to them, follow the instructions [here](./datasets/tac/Readme.md) for how to setup the data.

## Experiments
Calculating the proportion of ROUGE/BERTScore that can be explained by matches between SCUs (Section 4) can be calculated by running
```
sh experiments/scu-comparison/run.sh
```
The output plots will be in `experiments/scu-comparison/output/{tac2008,tac2009}/plots` and aggregate statistics in `experiments/scu-comparison/output/{tac2008,tac2009}/stats.json`.

The contributions of each category to the overall score, the contributions of each category type to the overall score, and the difference in categories between the two systems trained on CNN/DM can be calculated using:
```
sh experiments/metric-decomposition/{cnndm,tac}/run.sh
```
The respective directories will contain an `output` folder that contains the data output into `.tex` files which were used to create the tables in the paper.

The pairwise correlations between all of the metrics can be calculated using the [SacreROUGE](https://github.com/danieldeutsch/sacrerouge) library.
The scripts are not included here.

## Notes
Our experiment using BERTScore are based on [our fork](https://github.com/danieldeutsch/bert_score_content_analysis) of the [official repository](https://github.com/Tiiiger/bert_score), which includes code to return the alignment used by BERTScore.
The `requirements.txt` will automatically install our fork.
Because BERTScore aligns subword tokens, we aggreate the alignment at the token level.
Further, it automatically adds BOS and EOS tokens to the sequence.
Our processing code adds dummy tokens after calculating the alignment to account for this.