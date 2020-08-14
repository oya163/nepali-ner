"""
    Needs code structuring
    Date - 08/14/2020
"""

import torch
import logging
import sys
from flask import Flask, render_template, request
from utils.dataloader2 import Dataloader
from models.models import LSTMTagger
from config.config import Configuration

app = Flask(__name__)


def get_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(levelname)s:%(message)s"
    ))
    logging.getLogger().addHandler(handler)
    return logger


def get_config():
    config_file = "./config/config.ini"
    logger = get_logger()
    config = Configuration(config_file=config_file, logger=logger)
    config.model_file = './saved_models/lstm_1.pth'
    config.vocab_file = './vocab/vocab.pkl'
    config.label_file = './vocab/labels.pkl'
    config.device = 'cpu'
    config.verbose = False
    config.eval = False
    config.use_pos = False
    config.infer = True
    return config


def pred_to_tag(dataloader, predictions):
    return ' '.join([dataloader.label_field.vocab.itos[i] for i in predictions]).split()


def infer(config, dataloader, model):
    sent_tok = config.txt
    X = [dataloader.txt_field.vocab.stoi[t] for t in sent_tok]
    X = torch.LongTensor(X).to(config.device)
    X = X.unsqueeze(0)

    pred = model(X, None)
    pred_idx = torch.max(pred, 1)[1]

    y_pred_val = pred_idx.cpu().data.numpy().tolist()
    pred_tag = pred_to_tag(dataloader, y_pred_val)

    return pred_tag


# Inference section
def inference(config):
    dataloader = Dataloader(config, '1')
    model = LSTMTagger(config, dataloader).to(config.device)
    model.load_state_dict(torch.load(config.model_file)['state_dict'])
    pred_tag = infer(config, dataloader, model)
    return pred_tag


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/post', methods=['GET', 'POST'])
def post():
    config = get_config()
    errors = []
    text = request.form['input']
    config.txt = text.split()
    res = inference(config)
    results = zip(config.txt, res)

    if request.method == "GET":
        return render_template('index.html')
    else:
        return render_template('index.html', errors=errors, results=results)


if __name__ == "__main__":
    app.run()
