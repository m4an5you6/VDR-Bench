import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
'''%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True'''

from bert_score import score

with open("referrence.txt") as f:
    cands = [line.strip() for line in f]

with open("model_predict.txt") as f:
    refs = [line.strip() for line in f]
    

print(cands[0])
P, R, F1 = score(cands, refs, lang='en', verbose=True)

print(f"System level F1 score: {F1.mean():.3f}")
