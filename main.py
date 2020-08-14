#!/usr/bin/env python3
'''
    Main file
    Author: Oyesh Mann Singh
    
    How to run:
        python main.py -k 1 -d cpu
'''

import os
import argparse
import shutil
import warnings
from utils.dataloader import Dataloader
import utils.utilities as utilities
import utils.splitter as splitter
from tqdm import tqdm

from config.config import Configuration
from models.models import LSTMTagger, CharLSTMTagger
from train import Trainer

tqdm.pandas(desc='Progress')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="NER Main Parser")
    parser.add_argument("-c", "--config", dest="config_file", type=str, metavar="PATH", default="./config/config.ini",
                        help="Configuration file path")
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, metavar="PATH", default="./logs",
                        help="Log file path")
    parser.add_argument("-d", "--device", dest="device", type=str, default="cuda:3",
                        help="device[‘cpu’,‘cuda:0’,‘cuda:1’,..]")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, help="Print data description")
    parser.add_argument("-e", "--eval", action='store_true', default=False, help="For evaluation purpose only")
    parser.add_argument("-p", "--pos", action='store_true', default=False, help="Use POS one-hot-encoding")
    parser.add_argument("-r", "--char", action='store_true', default=False, help="Use character-level CNN")
    parser.add_argument("-g", "--grapheme", action='store_true', default=False, help="Use grapheme-level CNN")
    parser.add_argument("-k", "--kfold", dest="kfold", type=int, default=5, metavar="INT",
                        help="K-fold cross validation [default:1]")
    parser.add_argument("-i", "--infer", action='store_true',
                        default=False, help="For inference purpose only")
    parser.add_argument("--txt", dest="txt", type=str,
                        default="रबि लामिछाने नेपालि जन्ता को हिरो हुन", help="Input text (For inference purpose only)")

    args = parser.parse_args()
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.mkdir(args.log_dir)

    # Init Logger
    log_file = os.path.join(args.log_dir, 'complete.log')
    data_log = os.path.join(args.log_dir, 'data_log.log')
    logger = utilities.get_logger(log_file)

    config = Configuration(config_file=args.config_file, logger=logger)
    config.device = args.device
    config.verbose = args.verbose
    config.eval = args.eval
    config.kfold = args.kfold
    config.log_dir = args.log_dir
    config.log_file = log_file
    config.data_log = data_log
    config.use_pos = args.pos
    config.use_char = args.char
    config.use_graph = args.grapheme
    config.infer = args.infer
    config.txt = args.txt

    logger.info("***************************************")
    logger.info("Data file : {}".format(config.data_file))
    logger.info("Device : {}".format(config.device))
    logger.info("Verbose : {}".format(config.verbose))
    logger.info("Eval mode : {}".format(config.eval))
    logger.info("K-fold : {}".format(config.kfold))
    logger.info("Log directory: {}".format(config.log_dir))
    logger.info("Data log file: {}".format(config.data_log))
    logger.info("Use POS one-hot-encoding: {}".format(config.use_pos))
    logger.info("Use character-level CNN: {}".format(config.use_char))
    logger.info("Use grapheme-level CNN: {}".format(config.use_graph))
    logger.info("Inference mode: {}".format(config.infer))
    if config.infer:
        logger.info("Text: {}".format(config.txt))
    logger.info("***************************************")

    #     if not config.eval:
    #         if os.path.exists(config.output_dir):
    #             shutil.rmtree(config.output_dir)
    #         os.mkdir(config.output_dir)

    #         if os.path.exists(config.results_dir):
    #             shutil.rmtree(config.results_dir)
    #         os.mkdir(config.results_dir)

    return config, logger


# Inference section
def infer(config, logger):
    k = str(1)
    dataloader = Dataloader(config, k)

    # Load model
    arch = LSTMTagger(config, dataloader).to(config.device)

    # Print network configuration
    logger.info(arch)

    # Trainer
    model = Trainer(config, logger, dataloader, arch, k)

    model.load_checkpoint()

    logger.info("Inferred results")

    pred_tag = model.infer(config.txt)

    for s, p in zip(config.txt.split(), pred_tag):
        print(s + '\t' + p + '\n')

    return pred_tag


def train_test(config, logger):
    """
        Main File
    """
    if config.kfold > 0 and not config.eval:
        logger.info("Splitting dataset into {0}-fold".format(config.kfold))
        splitter.main(input_file=config.data_file,
                      output_dir=config.root_path,
                      verbose=config.verbose,
                      kfold=config.kfold,
                      pos=config.use_pos,
                      log_file=config.data_log)

    tot_acc = 0
    tot_prec = 0
    tot_rec = 0
    tot_f1 = 0

    for i in range(0, config.kfold):
        # To match the output filenames
        k = str(i + 1)

        if not config.eval:
            logger.info("Starting training on {0}th-fold".format(k))

        # Load data iterator
        dataloader = Dataloader(config, k)

        # Debugging purpose. Don't delete
        #         sample = next(iter(train_iter))
        #         print(sample.TEXT)

        # Load model
        if config.use_char or config.use_graph:
            assert config.use_char ^ config.use_graph, "Either use Character-Level or Grapheme-Level. Not both!!!"
            lstm = CharLSTMTagger(config, dataloader).to(config.device)
        else:
            lstm = LSTMTagger(config, dataloader).to(config.device)

        # Print network configuration
        logger.info(lstm)

        model = Trainer(config, logger, dataloader, lstm, k)

        if not config.eval:
            # Train
            logger.info("Training started !!!")
            model.fit()

        # Test
        model.load_checkpoint()
        logger.info("Testing Started !!!")
        acc, prec, rec, f1 = model.predict()
        logger.info("Accuracy: %6.2f%%; Precision: %6.2f%%; Recall: %6.2f%%; FB1: %6.2f " % (acc, prec, rec, f1))

        tot_acc += acc
        tot_prec += prec
        tot_rec += rec
        tot_f1 += f1

    logger.info("Final Accuracy: %6.2f%%; Final Precision: %6.2f%%; Final Recall: %6.2f%%; Final FB1: %6.2f " % (
        tot_acc / config.kfold, tot_prec / config.kfold, tot_rec / config.kfold, tot_f1 / config.kfold))


def main():
    """
        Main File
    """
    # Parse argument
    config, logger = parse_args()

    if config.infer:
        infer(config, logger)
    else:
        train_test(config, logger)


if __name__ == "__main__":
    main()
