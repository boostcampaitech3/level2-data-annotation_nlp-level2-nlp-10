import os
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from transformers import AutoModel
from load_data import *
import wandb
import torch.nn as nn
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# BiLSTM -> FC
class Model(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 10
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, (last_hidden, last_cell) = self.lstm(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output)
    # logits : (batch, num_labels)

    return {'logits' : logits}

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs= False):
        device= torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        labels= inputs.pop('labels')
        outputs= model(**inputs)
        
        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]

        custom_loss = torch.nn.CrossEntropyLoss.to(device)
        loss = custom_loss(outputs['logits'], labels)    
        return (loss, outputs) if return_outputs else loss
        

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'per:production', 'per:title', 'org:production', 'com:date_of_produced',
                  'com:sub_concept', 'com:alternative_names', 'com:made_of', 'com:prior_technology', 'com:similar_technology']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(10)[labels]
    score = np.zeros((10,))
    for c in range(10):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }


def label_to_num(label):
  num_label = []
  label_to_num = {'no_relation': 0, 'per:production': 1, 'per:title': 2, 'org:production': 3, 'com:date_of_produced': 4,
   'com:sub_concept': 5, 'com:alternative_names': 6, 'com:made_of': 7, 'com:prior_technology': 8, 'com:similar_technology': 9}
  for v in label:
    num_label.append(label_to_num[v])

  return num_label

def train():

  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[COM]", "[DAT]", "[PER]", "[ORG]", "[POH]"]})
  
  # load dataset
  train_dataset = load_data("./train.csv")

  train_label = label_to_num(train_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model =  Model(MODEL_NAME)
  model.model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
  model.to(device)

  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_strategy='no',
    save_total_limit=1,              # number of total save model.
    num_train_epochs=1,              # total number of training epochs
    learning_rate=3e-6,               # learning_rate
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_ratio = 0.1,
    weight_decay=0.01,               # strength of weight decay
    label_smoothing_factor=0.1,
    # lr_scheduler_type = 'cosine',
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='no', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    load_best_model_at_end = True,
    report_to = 'wandb',
    run_name = 'annotation_test'
  )

  trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    # eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  trainer.train()
  torch.save(model.state_dict(), os.path.join(f'./best_model', 'pytorch_model.bin'))

def main():
  train()

if __name__ == '__main__':
  wandb.init(project="ANNOTATION")
  wandb.run.name = f'annotation_test'
  seed_everything(42)
  # os.environ["WANDB_DISABLED"] = "true"
  
  main()