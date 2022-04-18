import pandas as pd

team = ['jeongho', 'jimin', 'kiwon', 'namhyeon', 'sun', 'wonsik']

sentence, subject_entity, object_entity, label = [], [], [], []
for t in team:
    csv = pd.read_csv("./" + t + ".csv")
    for s, se, oe, l in zip(csv['sentence'], csv['subject_entity'], csv['object_entity'], csv['label']):
        sentence.append(s)
        subject_entity.append(eval(se))
        object_entity.append(eval(oe))
        label.append(l)

output = pd.DataFrame(
    {'id': [i for i in range(len(sentence))], 'sentence': sentence, 'subject_entity': subject_entity, 'object_entity': object_entity, 'label': label})

output.to_csv('./output.csv', index=False)