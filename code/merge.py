import pandas as pd

team = ['jeongho', 'jimin', 'kiwon', 'namhyeon', 'sun', 'wonsik']
label2num = {'no_relation': 1, 'per:production': 2, 'per:title': 3, 'org:production': 4, 'com:date_of_produced': 5, 'com:date_of_prod': 5,
          'com:sub_concept': 6, 'com:alternative_names': 7, 'com:alter_names': 7, 'com:made_of': 8, 'com:prior_technology': 9, 'com:prior_tech': 9,
          'com:similar_technology': 10, 'com:similar_tech' : 10}
num2label = {1: 'no_relation', 2: 'per:production', 3: 'per:title', 4: 'org:production', 5: 'com:date_of_produced',
          6: 'com:sub_concept', 7: 'com:alternative_names', 8: 'com:made_of', 9: 'com:prior_technology',
          10: 'com:similar_technology'}

sentence, subject_entity, object_entity, label, worker1, worker2 = [], [], [], [], [], []
cnt = 0
for t in team:
    print(t)
    csv = pd.read_csv("./" + t + ".csv", encoding='cp949')
    for s, se, oe, l, w1, w2 in zip(csv['sentence'], csv['subject_entity'], csv['object_entity'], csv['label'], csv['worker1'], csv['worker2']):
        sentence.append(s)
        subject_entity.append(eval(se))
        object_entity.append(eval(oe))
        label.append(num2label[label2num[l]])
        worker1.append(num2label[label2num[w1]])
        worker2.append(num2label[label2num[w2]])
        if label2num[l] != label2num[w1] and label2num[w1] != label2num[w2] and label2num[l] != label2num[w2]:
            print(cnt, l, w1, w2)
        cnt += 1
output = pd.DataFrame(
    {'id': [i for i in range(len(sentence))], 'sentence': sentence, 'subject_entity': subject_entity,
     'object_entity': object_entity, 'worker1': label, 'worker2': worker1, 'worker3': worker2})

output.to_csv('./output.csv', index=False)