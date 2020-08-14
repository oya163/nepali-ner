"""
    Needs code structuring
    Date - 08/14/2020
"""

from flask import Flask, render_template, request
from utils.dataloader2 import Dataloader
import utils.utilities as utilities
from models.models import LSTMInferer
from config.config import Configuration
import torch

app = Flask(__name__)


def get_config():
    config_file = "./config/config.ini"
    log_file = './logs/complete.log'
    logger = utilities.get_logger(log_file)
    config = Configuration(config_file=config_file, logger=logger)
    config.model_file = './saved_models/lstm_1.pth'
    config.device = 'cpu'
    config.verbose = False
    config.eval = False
    config.use_pos = False
    return config


def pred_to_tag(dataloader, predictions):
    return ' '.join([dataloader.label_field.vocab.itos[i] for i in predictions]).split()


def infer(config, dataloader, model):
    """
    Prints the result
    """
    # Tokenize the sentence and aspect terms
    # print(sent.split())
    sent_tok = config.txt

    # Get index from vocab
    X = [dataloader.txt_field.vocab.stoi[t] for t in sent_tok]

    # Convert into torch and reshape into [batch, sent_length]
    X = torch.LongTensor(X).to(config.device)
    X = X.unsqueeze(0)

    # Get predictions
    pred = model(X, None)

    pred_idx = torch.max(pred, 1)[1]

    y_pred_val = pred_idx.cpu().data.numpy().tolist()
    pred_tag = pred_to_tag(dataloader, y_pred_val)
    return pred_tag


# Inference section
def inference(config):
    dataloader = Dataloader(config, '1')
    model = LSTMInferer(config).to(config.device)
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
