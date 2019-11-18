# Named Entity Recognition for Nepali Language

Code to reproduce [Named Entity Recognition for Nepali Language](https://arxiv.org/abs/1908.05828)

We publicly release Nepali NER Dataset

National Nepali Corpus is provided by [Bal Krishna Bal](http://ku.edu.np/cse/faculty/bal/ ), Professor, Kathmandu University.

Nepali sentences were collected from News Collection [here](https://github.com/sndsabin/Nepali-News-Classifier)

## Dataset statistics

Based on number of tokens for entities

| Entities        | EBIQUITY | ILPRL |
|-----------------|------|-------|
| PER             | 5059 | 262   |
| ORG             | 3811 | 180   |
| LOC             | 2313 | 273   |
| MISC            | 0    | 461   |
| Total sentences | 3606 | 548   |

## Results

These results are obtained using [conlleval] (https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt) tools

| Dataset                | EBIQUITY | ILPRL  |
|------------------------|----------|--------|
| Stanford CRF           | 75.160   | 56.250 |
| BiLSTM                 | 85.535   | 77.718 |
| BiLSTM + POS           | 84.235   | 81.963 |
| BiLSTM + CNN (C)       | 86.520   | 80.045 |
| BiLSTM + CNN (G)       | **86.893**   | 80.843 |
| BiLSTM + CNN (C) + POS | 84.970   | 81.860 |
| BiLSTM + CNN (G) + POS | 85.210   | **82.190** |

## Comparison

| Dataset                   | EBIQUITY | ILPRL  |
|---------------------------|----------|--------|
| Bam et al. SVM            | 66.26    | 46.26  |
| Ma and Hovy w/ glove      | 83.63    | 72.1   |
| Lample et al. w/ word2vec | 86.49    | 78.48  |
| BiLSTM + CNN (G)          | **86.893**   | 80.843 |
| BiLSTM + CNN (G) + POS    | 85.210   | **82.190** |

## Usage

To run 5-fold cross validation for BiLSTM + POS + Grapheme-level CNN model

    python main.py -k 5 -d cuda:0 -p -g


## Reference
- https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF


## Contact
- osingh1@umbc.edu


