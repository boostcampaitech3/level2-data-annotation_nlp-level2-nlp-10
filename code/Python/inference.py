from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from load_data import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
from train import *


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          )
    logits = outputs['logits']
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  origin_label = []
  num_to_label = {0: 'no_relation', 1: 'per:production', 2: 'per:title', 3: 'org:production', 4: 'com:date_of_produced',
                  5: 'com:sub_concept', 6: 'com:alternative_names', 7: 'com:made_of', 8: 'com:prior_technology', 9: 'com:similar_technology'}
  for v in label:
    origin_label.append(num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label


def main():
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # load tokenizer
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[COM]", "[DAT]", "[PER]", "[ORG]", "[POH]"]})

  model = Model(MODEL_NAME)
  model.model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
  state_dict = torch.load(os.path.join(f'./best_model', 'pytorch_model.bin'))

  model.load_state_dict(state_dict)
  model.to(device)
  
  ## load test datset
  test_dataset_dir = "../test.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  print('---- Finish! ----')
  
if __name__ == '__main__':
  main()
