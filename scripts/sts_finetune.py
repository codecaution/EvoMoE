from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr

roberta = RobertaModel.from_pretrained(
    '/apdcephfs/private_brendenliu/fairseq/outputs/2022-09-26/10-41-50/glue_log/cc100/_6_110000/sts_b_checkpoint/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='STS-B-bin'
)

roberta.cuda()
roberta.eval()
gold, pred = [], []
with open('glue_data/STS-B/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
        tokens = roberta.encode(sent1, sent2)
        features = roberta.extract_features(tokens)
        predictions = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features)
        gold.append(target)
        pred.append(predictions.item())

print('| Pearson: ', pearsonr(gold, pred))
