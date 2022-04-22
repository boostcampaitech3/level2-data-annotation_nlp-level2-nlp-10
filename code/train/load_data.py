import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# Typed entity marker(punct) to Subject/Object Entity and Sentence
def preprocessing_dataset_with_sentence1(dataset : pd.DataFrame):
  subject_entity = []
  object_entity = []
  sentence = []
  for SEN, SE, OE in zip( dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    S_WORD = eval(SE)['word']
    S_TYPE = eval(SE)['type']
    S_TEMP = ' '.join(['@', '*', '['+S_TYPE+']', '*', S_WORD, '@'])
    subject_entity.append(S_TEMP)

    O_WORD = eval(OE)['word']
    O_TYPE = eval(OE)['type']
    O_TEMP = ' '.join(['#', '^', '['+O_TYPE+']', '^', O_WORD, '#'])
    object_entity.append(O_TEMP)

    sentence.append(SEN.replace(S_WORD, S_TEMP).replace(O_WORD, O_TEMP))

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label']})
  return out_dataset

def preprocessing_dataset_with_sentence2(dataset : pd.DataFrame):
  subject_entity = []
  object_entity = []
  sentence = []
  for SEN, SE, OE in zip( dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    S_WORD = eval(SE)['word']
    S_TYPE = eval(SE)['type']
    S_TEMP = ' '.join(['@', '*', '['+S_TYPE+']', '*', S_WORD, '@'])
    subject_entity.append(S_TEMP)

    O_WORD = eval(OE)['word']
    O_TYPE = eval(OE)['type']
    O_TEMP = ' '.join(['#', '^', '['+O_TYPE+']', '^', O_WORD, '#'])
    object_entity.append(O_TEMP)

    sentence.append(SEN.replace(S_WORD, S_TEMP).replace(O_WORD, O_TEMP))

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'], 'answer':dataset['answer']})
  return out_dataset

def load_data(dataset_dir : str):
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset_with_sentence1(pd_dataset)

  return dataset

def load_test_data(dataset_dir : str):
  pd_dataset = pd.read_csv(dataset_dir, encoding='cp949')
  dataset = preprocessing_dataset_with_sentence2(pd_dataset)

  return dataset

def tokenized_dataset(dataset : pd.DataFrame, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = e01 + ' 과 ' + e02 + '의 관계'
    concat_entity.append(temp)
  
  tokenized_sentence = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=160,
      add_special_tokens=True,
      return_token_type_ids = False
      )
  
  return tokenized_sentence