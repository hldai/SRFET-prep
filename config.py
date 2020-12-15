from platform import platform
from os.path import join
import socket

MACHINE_NAME = socket.gethostname()
DATA_DIR = '/data/hldai/data/srfet-data'

LOG_DIR = join(DATA_DIR, 'log')

WIKI_FETEL_WORDVEC_FILE = join(DATA_DIR, 'res/enwiki-20151002-nef-wv-glv840B300d.pkl')

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

FIGER_FILES = {
    'test-mentions': join(DATA_DIR, 'figer-afet/figer-dfet-test-mentions.json'),
    'test-sents': join(DATA_DIR, 'figer-afet/figer-dfet-test-sents.json'),
    'test-srl': join(DATA_DIR, 'figer-afet/figer-test-sents-srl.txt'),
    'test-sents-dep': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-dep.txt'),
    'test-pos-tags': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-pos.txt'),
    'presample-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/enwiki20151002-anchor-figer-pre0_1'),
    'srl-train-data-prefix': join(DATA_DIR, 'weakdata/enwiki20151002-anchor-srl-pre0_1'),
    'vfet-train-data-prefix': join(DATA_DIR, 'weakdata/wiki20151002-anchor-srl16'),
    'type-vocab': join(DATA_DIR, 'figer-afet/figer-type-vocab.txt'),
}

BBN_FILES = {
    'test-mentions': join(DATA_DIR, 'bbn-afet/bbn-dfet-test-mentions.json'),
    'test-sents': join(DATA_DIR, 'bbn-afet/bbn-dfet-test-sents.json'),
    'test-srl': join(DATA_DIR, 'bbn-afet/bbn-test-sents-srl.txt'),
    'test-sents-dep': join(DATA_DIR, 'bbn-afet/bbn-test-sents-tok-texts-dep.txt'),
    'test-pos-tags': join(DATA_DIR, 'bbn-afet/bbn-test-sents-tok-texts-pos.txt'),
    'srl-train-data-prefix': join(DATA_DIR, 'weakdata/wiki20151002-anchor-srl16'),
    'type-vocab': join(DATA_DIR, 'bbn-afet/bbn-type-vocab.txt'),
}
