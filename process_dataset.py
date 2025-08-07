import json
import random
from random import sample
import numpy as np


def get_demonstrations(input_files,train_sample_ids):
    length_n=[]
    length_l=[]
    length_m=[]
    length_h=[]
    length_H = []

    index_n=[]
    index_l=[]
    index_m = []
    index_h=[]
    index_H=[]

    with open(input_files, "r", encoding='utf-8-sig') as f:
        content = json.load(f)
        for i in train_sample_ids:
            if "Final Risk Level: 'n'" in content[i]['response']:
                index_n.append(i)
                length_n.append(len(content[i]['input']))
            elif "Final Risk Level: 'l'" in content[i]['response']:
                index_l.append(i)
                length_l.append(len(content[i]['input']))
            elif "Final Risk Level: 'm'" in content[i]['response']:
                index_m.append(i)
                length_m.append(len(content[i]['input']))
            elif "Final Risk Level: 'h'" in content[i]['response']:
                index_h.append(i)
                length_h.append(len(content[i]['input']))
            elif "Final Risk Level: 'H'" in content[i]['response']:
                index_H.append(i)
                length_H.append(len(content[i]['input']))

    n_id= index_n[length_n.index(min(length_n))]
    l_id = index_l[length_l.index(min(length_l))]
    m_id = index_m[length_m.index(min(length_m))]
    h_id = index_h[length_h.index(min(length_h))]
    H_id = index_H[length_H.index(min(length_H))]

    ids=[n_id,l_id,m_id,h_id,H_id]
    risk=['No risk','Low risk','Medium risk','High risk','Extremely high risk']
    demonstrations=str()
    with open(input_files, "r", encoding='utf-8-sig') as f:
        content = json.load(f)
        for i in ids:
            if i ==H_id:
                demonstrations+=f'Scene: ({content[i]["input"]}) Final risk level: {risk[ids.index(i)]}'
            else:
                demonstrations+=f'Scene: ({content[i]["input"]}) Final risk level: {risk[ids.index(i)]}\n'

    return demonstrations


def trans_data(input_file,method,test_sample_id,test_sample_num):


    output_file_train=f"./scene_train_{method}_{test_sample_num}.json"
    output_file_test =f"./scene_test_{method}_{test_sample_num}.json"
    with open(output_file_train, 'a+') as f_train:
        f_train.write('[\n')

    with open(output_file_test, 'a+') as f_test:
        f_test.write('[\n')


    with open(input_file, "r", encoding='utf-8-sig') as f1:
        content = json.load(f1)
        scene_list=content["rasa_nlu_data"]["common_examples"]
        sample_id=np.arange(len(scene_list)).tolist()
        train_sample_id=list(set(sample_id)-set(test_sample_id))

        if method=="process_step_by_step_normal":
            ##描述过程监督的第二种数据格式
            for i in range(len(scene_list)):
                one_data = dict()
                response = str()
                starts = []
                entry_list = scene_list[i]["entities"]
                for j in range(len(entry_list)):
                    starts.append(entry_list[j]["start"])
                sorted_starts = sorted(starts)
                for j in range(len(sorted_starts)):
                    if entry_list[starts.index(sorted_starts[j])]["entity"] == "positive":
                        response += "Aspect: " + entry_list[starts.index(sorted_starts[j])]["value"].strip().strip(
                            ",").strip(".") + '. ' + "Impact: Positive" + "\n"
                    if entry_list[starts.index(sorted_starts[j])]["entity"] == "neutral":
                        response += "Aspect: " + entry_list[starts.index(sorted_starts[j])]["value"].strip().strip(
                            ",").strip(".") + '. ' + "Impact: Neutral" + "\n"
                    if entry_list[starts.index(sorted_starts[j])]["entity"] == "negative":
                        response += "Aspect: " + entry_list[starts.index(sorted_starts[j])]["value"].strip().strip(
                            ",").strip(".") + '. ' + "Impact: Negative" + "\n"
                risk_level = scene_list[i]["intent"]
                if risk_level == "no risk":
                    response = response + "Final Risk Level: No risk"
                elif risk_level == "low risk":
                    response = response + "Final Risk Level: Low risk"
                elif risk_level == "medium risk":
                    response = response + "Final Risk Level: Medium risk"
                elif risk_level == "high risk":
                    response = response + "Final Risk Level: High risk"
                elif risk_level == "extremely high risk":
                    response = response + "Final Risk Level: Extremely high risk"

                one_data['instruction']="Please perform Scene Safety Level Recognition task. Given the paragraph, assign a safety level label from [No risk, Low risk, Medium risk, High risk, Extremely high risk]."
                one_data["input"] = scene_list[i]["text"]
                one_data["response"] = response

                if i in train_sample_id:
                    with open(output_file_train, 'a+') as f:
                        json.dump(one_data, f, indent=4)
                        if i == max(train_sample_id):
                            f.write('\n]')
                        else:
                            f.write(",\n")
                elif i in test_sample_id:
                    with open(output_file_test, 'a+') as f:
                        json.dump(one_data, f, indent=4)
                        if i == max(test_sample_id):
                            f.write('\n]')
                        else:
                            f.write(",\n")


        elif method=="final_label":
            for i in range(len(scene_list)):
                one_data = dict()
                one_data["instruction"] = scene_list[i]["text"]
                one_data["instruction"] = "Please perform Scene Safety Level Recognition task. Given the paragraph, assign a safety level label from [No risk, Low risk, Medium risk, High risk, Extremely high risk]. Return label only without any other text."

                risk_level = scene_list[i]["intent"]
                if risk_level == 'no risk':
                    one_data["response"] = 'No risk'
                elif risk_level == 'low risk':
                    one_data["response"] = 'Low risk'
                elif risk_level == 'medium risk':
                    one_data["response"] = 'Medium risk'
                elif risk_level == 'high risk':
                    one_data["response"] = 'High risk'
                elif risk_level == 'extremely high risk':
                    one_data["response"] = 'Extremely high risk'

                if i in train_sample_id:
                    with open(output_file_train, 'a+') as f:
                        json.dump(one_data, f, indent=4)
                        if i==max(train_sample_id):
                            f.write('\n]')
                        else:
                            f.write(",\n")
                elif i in test_sample_id:
                    with open(output_file_test, 'a+') as f:
                        json.dump(one_data, f, indent=4)
                        if i==max(test_sample_id):
                            f.write('\n]')
                        else:
                            f.write(",\n")

def split_train_test(input_file,num):
    #num表示测试集中需要的样本个数
    n_index=[]
    l_index=[]
    m_index=[]
    h_index=[]
    H_index=[]
    with open(input_file, "r", encoding='utf-8-sig') as f:
        content = json.load(f)
        scene_list=content["rasa_nlu_data"]["common_examples"]
        for i in range(len(scene_list)):
            risk_level = scene_list[i]["intent"]
            if risk_level=='no risk':
                n_index.append(i)
            elif risk_level=='low risk':
                l_index.append(i)
            elif risk_level=='medium risk':
                m_index.append(i)
            elif risk_level=='high risk':
                h_index.append(i)
            elif risk_level=='extremely high risk':
                H_index.append(i)

    random.seed(8)
    n_choice=sample(n_index,int(num/5))
    l_choice = sample(l_index, int(num/5))
    m_choice = sample(m_index, int(num/5))
    h_choice = sample(h_index,int(num/5))
    H_choice = sample(H_index, int(num/5))

    n_index_remain=list(set(n_index) - set(n_choice))
    l_index_remain = list(set(l_index) - set(l_choice))
    m_index_remain = list(set(m_index) - set(m_choice))
    h_index_remain = list(set(h_index) - set(h_choice))
    H_index_remain = list(set(H_index) - set(H_choice))

    return n_choice+l_choice+m_choice+h_choice+H_choice,[n_index_remain,l_index_remain,m_index_remain,h_index_remain,H_index_remain]

original_dataset_path='./scene_safety_all.json'
method='process_step_by_step_normal'
test_sample_num=705
test_sample_id,_ =split_train_test(original_dataset_path,test_sample_num)
trans_data(original_dataset_path,method,test_sample_id,test_sample_num)
