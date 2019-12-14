# Named Entity Recognition for Nepali Language

Code to reproduce [Named Entity Recognition for Nepali Language](https://arxiv.org/abs/1908.05828)

We publicly release Nepali NER Dataset version 1 and version 2. We have named this dataset as EBIQUITY as we published this paper/dataset while working in EBIQUITY lab in UMBC. They are further divided into raw and stemmed (brute-force approach) version.

* v1 - IO tagging scheme
* v2 - BIO tagging scheme with corrections. Correction details are stated in README.txt inside the dataset folder. **Recommended to use**

National Nepali Corpus can be found [here](https://www.sketchengine.eu/nepali-national-corpus/)

Nepali sentences were collected from online news website of the year [2015-2016](https://github.com/sndsabin/Nepali-News-Classifier) and [2009-2010](https://pdfs.semanticscholar.org/c8c4/d371c9b8a759b3927de6c2b0f1fa98f4501c.pdf)

## Dataset statistics

Based on number of tokens for entities

| Entities        | EBIQUITY | ILPRL |
|-----------------|------|-------|
| PER             | 5059 | 262   |
| ORG             | 3811 | 180   |
| LOC             | 2313 | 273   |
| MISC            | 0    | 461   |
| Total sentences | 3606 | 548   |

## Embedding comparison
| Embeddings          | Raw       | Stemmed |
|---------------------|-----------|---------|
| Random              | 73.98     | 76.410  |
| Word2Vec_CBOW       | 74.465    | 82.230  |
| Word2Vec_Skip Gram  | 76.873    | 84.330  |
| GloVe               | 75.718    | 83.833  |
| fastText_Pretrained | 80.403    | 82.068  |
| fastText_CBOW       | 78.343    | 81.415  |
| fastText_Skip Gram  | **81.793**    | **85.535**  |

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


