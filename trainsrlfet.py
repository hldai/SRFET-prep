import os
import datetime
import torch
import logging
import argparse
from exp import srlfetexp, expdata
from utils.loggingutils import init_universal_logging
import config


def __train3():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    margin = 1.0
    train_config = srlfetexp.TrainConfig(
        pos_margin=margin, neg_margin=margin, neg_scale=1.0, batch_size=128, schedule_lr=True)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl3-{}'.format(dataset))

    val_mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
    val_sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
    val_srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
    val_dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')
    val_manual_label_file = os.path.join(config.DATA_DIR, 'figer/figer-dev-man-labeled.txt')
    # manual_val_file_tup = (val_mentions_file, val_sents_file, val_dep_file, val_srl_file, val_manual_label_file)
    manual_val_file_tup = None

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(
        device, gres, train_data_pkl, dev_data_pkl, manual_val_file_tup, test_file_tup, lstm_dim, mlp_hidden_dim,
        type_embed_dim, train_config, single_type_path, save_model_file_prefix=save_model_file_prefix)


def __train2():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    margin = 2.0
    train_config = srlfetexp.TrainConfig(pos_margin=margin, neg_margin=margin, neg_scale=1.0, batch_size=128)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))

    val_mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
    val_sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
    val_srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
    val_dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')
    val_manual_label_file = os.path.join(config.DATA_DIR, 'figer/figer-dev-man-labeled.txt')
    # manual_val_file_tup = (val_mentions_file, val_sents_file, val_dep_file, val_srl_file, val_manual_label_file)
    manual_val_file_tup = None

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(
        device, gres, train_data_pkl, dev_data_pkl, manual_val_file_tup, test_file_tup, lstm_dim, mlp_hidden_dim,
        type_embed_dim, train_config, single_type_path, save_model_file_prefix=save_model_file_prefix)


def __train1():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    margin = 1.0
    train_config = srlfetexp.TrainConfig(pos_margin=margin, neg_margin=margin, neg_scale=1.0)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))

    val_mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
    val_sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
    val_srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
    val_dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')
    val_manual_label_file = os.path.join(config.DATA_DIR, 'figer/figer-dev-man-labeled.txt')
    manual_val_file_tup = (val_mentions_file, val_sents_file, val_dep_file, val_srl_file, val_manual_label_file)

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(
        device, gres, train_data_pkl, dev_data_pkl, manual_val_file_tup, test_file_tup, lstm_dim, mlp_hidden_dim,
        type_embed_dim, train_config, single_type_path, save_model_file_prefix=save_model_file_prefix)


def __train():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    train_config = srlfetexp.TrainConfig(loss_name='mm', neg_scale=0.1)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    # dataset = 'figer'
    dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(device, gres, train_data_pkl, dev_data_pkl, test_file_tup, lstm_dim, mlp_hidden_dim,
                           type_embed_dim, train_config, single_type_path,
                           save_model_file_prefix=save_model_file_prefix)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    args = parser.parse_args()

    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

    if args.idx == 0:
        __train()
    if args.idx == 1:
        __train1()
    if args.idx == 2:
        __train2()
    if args.idx == 3:
        __train3()
