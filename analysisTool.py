"""Official evaluation script for SQuAD version 2.0.
In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""

import collections
from collections import Counter
import json
import numpy as np
import os
import re
import string
import sys
import pandas as pd
import matplotlib.pyplot as plt 

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def plot_pr_curve(precisions, recalls, out_image, title):
  
  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()

def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i+1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  if out_image:
    plot_pr_curve(precisions, recalls, out_image, title)
  
  return {'ap': 100.0 * avg_prec}

def run_precision_recall_analysis(exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir):
  if out_image_dir and not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)
  num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
  if num_true_pos == 0:
    return
  pr_exact = make_precision_recall_eval(
      exact_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_exact.png'),
      title='Precision-Recall curve for Exact Match score')
  pr_f1 = make_precision_recall_eval(
      f1_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_f1.png'),
      title='Precision-Recall curve for F1 score')
  oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
  pr_oracle = make_precision_recall_eval(
      oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
      title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')


def histogram_na_prob(na_probs, qid_list, image_dir, name):
  if not qid_list:
    return
  x = [na_probs[k] for k in qid_list]
  weights = np.ones_like(x) / float(len(x))
  plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
  plt.xlabel('Model probability of no-answer')
  plt.ylabel('Proportion of dataset')
  plt.title('Histogram of no-answer probability: %s' % name)
  plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
  plt.clf()




def collect_ndata(ndata,dataset,preds,filename_ndata):
  ndata_id = ndata['id'].values.tolist()
  data = {
    'id':[],
    'context':[],
    'question':[],
    'answer':[],
    'predicted_answer':[],
    'f1':[],
    'em':[],
    'precision':[],
    'recall':[]
  }
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        if qa['id'] in ndata_id and qa['id'] not in data['id']:
          data['id'].append(qa['id'])
          data['context'].append(p['context'])
          data['question'].append(qa['question'])
          data['answer'].append(qa['answers'])
          data['predicted_answer'].append(preds[qa['id']])
          data['f1'].append(ndata[ndata.id==qa['id']].f1.values[0])
          data['em'].append(ndata[ndata.id==qa['id']].em.values[0])
          data['precision'].append(ndata[ndata.id==qa['id']].precision.values[0])
          data['recall'].append(ndata[ndata.id==qa['id']].recall.values[0])
          
  
  # print(data)
  ndata_dataframe = pd.DataFrame(data,columns=['id','context','question','answer','predicted_answer','f1','em','precision','recall'])
  # print(ndata_dataframe)
  ndata_dataframe.sort_values(by=['f1'], inplace=True,ascending=True)
  # ndata_dataframe.reset_index()
  print(ndata_dataframe.head())
  ndata_dataframe.to_json('./analysis/'+filename_ndata+'.json',orient="table",index=False)
  


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def prec(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return precision 

def recall(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    em = {}
    fscore = {}
    ids = []
    precision = []
    recalls = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]

                ################################################################################
                
                ids.append(qa['id'])

                # em.append(metric_max_over_ground_truths(
                #     exact_match_score, prediction, ground_truths))
                
                em[qa['id']]=metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                # fscore.append(metric_max_over_ground_truths(
                #     f1_score, prediction, ground_truths))
                
                fscore[qa['id']]=metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)


                precision.append(metric_max_over_ground_truths(
                    prec, prediction, ground_truths))
                
                recalls.append(metric_max_over_ground_truths(
                    recall, prediction, ground_truths))

                ################################################################################

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return ids,f1,exact_match,em,fscore,precision,recalls

def fire(dataset_file,preds_file):
  with open('./uploads/'+dataset_file) as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']
  with open('./uploads/'+preds_file) as f:
    preds = json.load(f)

  na_probs = {k: 0.0 for k in preds}
  qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  ids,f1,exact_match,em,fscore,precision,recalls = evaluate(dataset, preds)
  
  dataframe = pd.DataFrame({
    'id':ids,
    'em':em.values(),
    'f1':fscore.values(),
    'precision':precision,
    'recall':recalls
  })
  # print(f1,exact_match)
  # print(dataframe)
  # dataframe.sort_values(by=['f1'], inplace=True)
  # dataframe.to_json('eval.json')

  out_eval = make_eval_dict(em, fscore)

  run_precision_recall_analysis(em, fscore, na_probs, 
                                  qid_to_has_ans, './images')

  histogram_na_prob(na_probs, has_ans_qids, './images', 'hasAns')

  histogram_na_prob(na_probs, no_ans_qids, './images', 'noAns')

  print(json.dumps(out_eval, indent=2))

  dataframe.sort_values(by=['f1'], inplace=True,ascending=True)

  nworst = dataframe.head(20)

  nbest = dataframe.tail(10)
  
  collect_ndata(nworst,dataset,preds,'nworst')
  
  collect_ndata(nbest,dataset,preds,'nbest')

  collect_ndata(dataframe,dataset,preds,'all')
  return out_eval

# if __name__ == '__main__':
#   fire('dataset.json','pred_data.json')
 