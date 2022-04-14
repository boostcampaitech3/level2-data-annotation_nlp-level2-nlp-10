import os
import requests
import json
import html
import zipfile
import shutil

url = "https://tagtog.net/-login"
file_url = '???' # Download All Documents 버튼 링크 주소
o_file = 'abc.zip'
if os.path.exists(o_file):
    os.remove(o_file)

login_info = {
    'loginid' : '???', # tagtog 로그인 이메일
    'password' : '???', # 비밀번호
}

with requests.Session() as s:
    login_req = s.post(url, data=login_info)
    r = s.get(file_url)

    with open(o_file,"wb") as output:
        output.write(r.content)

folder_path = "./tagtog_relation_extraction/"
zip_ = zipfile.ZipFile("abc.zip")
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
zip_.extractall(folder_path)

folder_name = "./tagtog_relation_extraction/???/" # 자신의 프로젝트명
context_name_list = os.listdir(folder_name + "ann.json/master/pool")
relation_folder_paths = [folder_name + "ann.json/master/pool/"]
contexts_folders_paths = [folder_name + "plain.html/pool/"]
annotation_legend = folder_name + "annotations-legend.json"
with open(annotation_legend,"r") as f:
    annotation_legend = json.load(f)


def get_context_from_html(html_file):
    import re
    html_file = re.sub(r"\n", " ", html_file)
    html_file = html.unescape(html_file)  # 21-11-17 추가, &quot; 등 제거
    return re.findall("(<pre.+>)(.+)(</pre>)", html_file)[0][1]


id = []
sentence = []
subject_entity = []
object_entity = []
label = []
count = 0
for context_name, relation_folder, contexts_folder in zip(context_name_list, relation_folder_paths, contexts_folders_paths):
    file_ids = [file_name.split(".txt.")[0] for file_name in os.listdir(relation_folder)]
    file_nums = [ids.split("-")[1] for ids in file_ids]
    relation_files = [relation_folder + file_id + ".txt.ann.json" for file_id in file_ids]
    context_files = [contexts_folder + file_id + ".txt.plain.html" for file_id in file_ids]

    for relation_file, context_file, file_num in zip(relation_files, context_files, file_nums):
        with open(relation_file, "r", encoding='UTF-8') as f:
            relation_json = json.load(f)

        with open(context_file, "r", encoding='UTF-8') as f:
            context_json = f.read()

        tmp_sentence = get_context_from_html(context_json)

        for r in relation_json['relations']:
            tmp_sub_entity, tmp_obj_entity = {}, {}
            for re in r['entities']:
                _, entity, start_end = re.split('|')
                entity = annotation_legend[entity].split('-')
                start, end = map(int, start_end.split(','))
                if entity[0] == 'SUB':
                    tmp_sub_entity['word'] = tmp_sentence[start:end+1]
                    tmp_sub_entity['start'] = start
                    tmp_sub_entity['end'] = end
                    tmp_sub_entity['type'] = entity[1]
                else:
                    tmp_obj_entity['word'] = tmp_sentence[start:end + 1]
                    tmp_obj_entity['start'] = start
                    tmp_obj_entity['end'] = end
                    tmp_obj_entity['type'] = entity[1]

            if len(entity) == 2: tmp_label = 'no_relation'
            else: tmp_label = (tmp_sub_entity['type'] + ':' + entity[2]).lower()

            id.append(count)
            count += 1
            sentence.append(tmp_sentence)
            subject_entity.append(tmp_sub_entity)
            object_entity.append(tmp_obj_entity)
            label.append(tmp_label)

print(id)
print(sentence)
print(subject_entity)
print(object_entity)
print(label)

import pandas as pd
output = pd.DataFrame({'id': id, 'sentence': sentence, 'subject_entity': subject_entity, 'object_entity': object_entity, 'label': label})
output.to_csv('./output.csv', index=False)