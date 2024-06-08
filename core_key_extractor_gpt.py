import json
import requests
from time import sleep
from utils import * 
from random import shuffle 
from openai import OpenAI
import pandas as pd 



def write_prompts():
	prompt_dict = {}
	keys = ['prompt_tmp', 'cat_prompt_tmp', 'prompt_tmp_msqa', 'cat_prompt_tmp_msqa', 'prompt_tmp_techqa', 'cat_prompt_tmp_techqa']
	arr = [prompt_tmp, cat_prompt_tmp, prompt_tmp_msqa, cat_prompt_tmp_msqa, prompt_tmp_techqa, cat_prompt_tmp_techqa]
	for idx,pmt in enumerate(arr):
		prompt_dict[keys[idx]] = pmt

	with open('prompts.json', 'w') as f:
		json.dump(prompt_dict, f)

def read_prompts():
	
	with open('prompts.json') as f:
		prompt_dict = json.load(f)
	return prompt_dict

def read_techqa():
	techqa_df = pd.read_excel(techqa_path)
	with open(techqa_ans_path) as f:
		techqa_ans_data = json.load(f)

	techqa_ans = {}

	for item in techqa_ans_data:
		techqa_ans[item['QUESTION_TEXT']] = item['ANSWER']

	filtered_techqa = {}

	for idx, row in techqa_df.iterrows():
		if row['Path to highligted ground truth'] != '' and type(row['Path to highligted ground truth']) != float and type(row['question']) != float:
			q = row['question'].replace('_x000D_','\r') 
			ans = techqa_ans[q]
			filtered_techqa[q] = [row['title'], ans, row['Question Type -- Bhavya']]
			print(q, filtered_techqa[q])
	return filtered_techqa

def read_msqa(path, n):
	with open(path) as f:
		data = json.load(f)
	msqa_q = {}
	cnt = 0
	for row in data:
		if row['label'] not in ["['other']", "[]"]:
			if cnt == n:
				break 
			cnt += 1
			msqa_q[row['text']] = [row['answer'], row['label']]
	return msqa_q 


def prompt_gpt(payload):
	api_key = API_KEY
	client = OpenAI(api_key=api_key)
	model = 'gpt-4-1106-preview'
	# model = 'gpt-3.5-turbo'
	_response = client.chat.completions.create(
  model=model,
  messages=[{
      "role": "system",
      "content": "You are a helpful biomedical assistant expert at identifying the core information from answers. Please respond precisely and concisely."
    },{"role": "user", "content": payload}],
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    logprobs=True,
                    n=1
                )

	

    # print(_response['choices'])

	print(_response.choices[0].message.content)
	return _response.choices[0].message.content

def prompt_hf(payload, API_URL, headers):
	json_body = {
		"inputs": payload,
				"parameters": {"max_new_tokens":2000, "top_k":40, "top_p":0.1,"repetition_penalty":1.176, "temperature":0.7},
				"wait_for_model":True
		}
	data = json.dumps(json_body)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	try:
		return json.loads(response.content.decode("utf-8"))[0]['generated_text']
	except:
		print(response.content)
		return response

def extract_msqa():
	# msqa = read_msqa(msqa_paths[0], 1)
	# msqa_q2 = read_msqa(msqa_paths[1], 30)
	# msqa_q3 = read_msqa(msqa_paths[2], 30)
	# msqa_q4 = read_msqa(msqa_paths[3], 45)
	# msqa.update(msqa_q2)
	# msqa.update(msqa_q3)
	# msqa.update(msqa_q4)

	with open(idir+'.xlsx','rb') as f:
		df = pd.read_excel(f)

	odict = {}
	# for key, info in msqa.items():
	for idx, row in df.iterrows():
		sleep(1)
		q = row['question'].replace('\n','\t')
		ans = row['ideal answer'].replace('\n','\t')
		cat = row['question type'].lower()[2:-2]
		print(cat)
		if cat == 'root cause':
			cat = 'resolution'
		# prompt = cat_prompt_tmp_msqa[cat].replace('{{ques}}',q).replace('{{ans}}',ans.replace('\n','\t'))
		prompt = prompt_tmp_msqa.replace('{{ques}}',key.replace('\n','\t')).replace('{{ans}}',info[0].replace('\n','\t'))
		print(prompt)
		output = prompt_gpt(prompt)
		print(output)
		odict[q] = output
		# try:
		with open(odir+'.json','w') as f:
			json.dump(odict,f)
	olst = []
	# for key, info in msqa.items():
	for idx,row in df.iterrows():
		olst.append([row['question'], odict[row['question'].replace('\n','\t')], row['ideal answer'], row['question type']])
		# olst.append([key, odict[key], info[0], info[1]])
	
	df = pd.DataFrame(olst)
	df.columns = ['question', 'pred exact ans', 'ideal answer', 'question type']
	df.to_excel(odir+'.xlsx')


def extract_msqa2():
	with open(msqa_labeled_path,'rb') as f:
		df = pd.read_excel(f)

	# with open(msqa_mapping) as f:
	# 	mapping = json.load(f)

	# mapping_rev = {}
	# for k,v in mapping.items():
	# 	mapping_rev[v] = k 

	odict = {}
	# for key, info in msqa.items():
	for idx, row in df.iterrows():
		# if row['question'] in mapping_rev:
		if row['question'].strip() == "How to create partition on an existing non-partitioned table in synapse dedicated sql pool?":
			sleep(1)
			q = row['question'].replace('\n','\t')
			ans = row['ideal answer'].replace('\n','\t')
			cat = row['question type'].lower()[:-1]
			print(cat)
			if cat == 'root cause':
				cat = 'resolution'
			prompt = cat_prompt_tmp_msqa[cat].replace('{{ques}}',q).replace('{{ans}}',ans.replace('\n','\t'))
			# prompt = prompt_tmp_msqa.replace('{{ques}}',q.replace('\n','\t')).replace('{{ans}}',ans.replace('\n','\t'))
			print(prompt)
			output = prompt_gpt(prompt)
			print(output)
			odict[q] = output
			# try:
		
	olst = []
	# for key, info in msqa.items():
	for idx,row in df.iterrows():
		# if row['question'] in mapping_rev:
		if row['question'].strip() == "How to create partition on an existing non-partitioned table in synapse dedicated sql pool?":
			olst.append([row['question'], odict[row['question'].replace('\n','\t')], row['ideal answer'], row['question type']])
		# olst.append([key, odict[key], info[0], info[1]])
	
	df = pd.DataFrame(olst)
	df.columns = ['question', 'pred exact ans', 'ideal answer', 'question type']
	df.to_excel(msqa_out_path)


def extract_techqa():
	filtered_techqa = read_techqa()

	odict = {}
	for key, info in filtered_techqa.items():
		sleep(1)
		q = (info[0] + '. ' + key).replace('\n','\t')
		# prompt = prompt_tmp_techqa.replace('{{ques}}',q).replace('{{ans}}',info[1])
		prompt = cat_prompt_tmp_techqa[info[2].strip()].replace('{{ques}}',q).replace('{{ans}}',info[1])
		print(prompt)
		output = prompt_gpt(prompt)
		print(output)
		odict[q] = output
		# try:
		with open(odir+'.json','w') as f:
			json.dump(odict,f)
	olst = []
	for key, info in filtered_techqa.items():
		olst.append([info[0], key, odict[info[0] + '. ' + key.replace('\n','\t')], info[1], info[2]])
	df = pd.DataFrame(olst)
	df.columns = ['title', 'question', 'pred exact ans', 'ideal answer', 'question type']
	df.to_excel(odir+'.xlsx')

def extract_techqa2():
	with open(techqa_labeled_path,'rb') as f:
		df = pd.read_excel(f)

	with open(techqa_mapping) as f:
		mapping = json.load(f)

	mapping_rev = {}
	for k,v in mapping.items():
		mapping_rev[v] = k 


	odict = {}
	for idx, row in df.iterrows():
		q = row['title'] + '. ' + row['question'].replace('\n','\t')
		if q in mapping_rev:
			sleep(1)
			prompt = prompt_tmp_techqa.replace('{{ques}}',q).replace('{{ans}}',row['ideal answer'].replace('\n','\t'))
			
			# prompt = cat_prompt_tmp_techqa[row['question type'].strip()].replace('{{ques}}',q).replace('{{ans}}',row['ideal answer'].replace('\n','\t'))
			print(prompt)
			output = prompt_gpt(prompt)
			print(output)
			odict[q] = output
			
	olst = []
	for idx,row in df.iterrows():
		q = row['title'] + '. ' + row['question'].replace('\n','\t')
		if q in mapping_rev:
			olst.append([row['title'], row['question'], odict[q], row['ideal answer'], row['question type']])
	df = pd.DataFrame(olst)
	df.columns = ['title', 'question', 'pred exact ans', 'ideal answer', 'question type']
	df.to_excel(techqa_out_path)

def complete_answer_bioasq(inpaths, outpath, pred_path):
	all_ques = []
	all_cats = []
	all_ans = []
	all_exact = []
	for path in inpaths['bioasq']:
		ques, cats, ans_id, ans_ex = read_bioasq(path)
		for idx, q in enumerate(ques):
			if cats[idx] != 'summary':
				all_ques.append(q)
				all_cats.append(cats[idx])
				all_ans.append(ans_id[idx])
				all_exact.append(ans_ex[idx])

	prompt_dict = read_prompts()

	all_pred_exact_dict = {}
	for path in pred_path:
		with open(path) as f:
			all_pred_exact_dict.update(json.load(f))

	all_pred_exact = []
	for q in all_ques:
		all_pred_exact.append(all_pred_exact_dict[q])

	indices = list(range(len(all_ques)))
	shuffle(indices)
	sel_idx = indices[:10]

	odict = {}

	for idx, q in enumerate(all_ques):
		if idx in sel_idx:
			sleep(1)
			prompt = prompt_dict['prompt_tmp_bioasq_complete'].replace('{{ques}}',q)
			print(prompt)
			output = prompt_gpt(prompt)
			print(output)
			odict[q] = output
			
	olst = []
	for idx,q in enumerate(all_ques):
		if idx in sel_idx:
			olst.append([q, odict[q], all_ans[idx],  all_exact[idx], all_pred_exact[idx], all_cats[idx]])
	df = pd.DataFrame(olst)
	df.columns = ['question', 'pred complete ans', 'ideal answer',  'true exact ans', 'pred exact ans','question type']
	df.to_excel(outpath, index=False)


def extract_bioasq(inpath, outpath, model_name,API_URL,headers, system = False):
	all_ques = []
	all_cats = []
	all_ans = []
	
	
	ques, cats, ans_id, ans_ex = read_bioasq(inpath)
	for idx, q in enumerate(ques):
		if cats[idx] != 'summary':
			all_ques.append(q)
			all_cats.append(cats[idx])
			all_ans.append(ans_id[idx])
			
	prompt_dict = read_prompts()

	print(len(all_ques), len(all_ans))

	indices = list(range(len(all_ques)))
	# shuffle(indices)
	sel_idx = indices

	odict = {}

	for idx, q in enumerate(all_ques):

		if idx in sel_idx:
			sleep(0.5)
		
			if type(all_ans[idx]) == list:
				idans = ' '.join(all_ans[idx])
			else:
				idans = all_ans[idx]
			prompt = prompt_dict['prompt_tmp'].replace('{{ques}}',q.replace('\n','\t')).replace('{{ans}}',idans.replace('\n','\t'))
			
			examples = prompt.split('\n====\nQuestion:')
			qs = []
			ans_lst = []
			# print(examples)
			print('---'*20)
			for ex in examples:
				iq, ans = ex.split('Exact Answer:')
				qs.append(iq)
				ans_lst.append(ans)

			if 'llama' in outpath:
				if system:
					prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful biomedical assistant expert at identifying the core information from answers.  Please respond precisely and concisely. <|eot_id|>"
				else:
					prompt = "<|begin_of_text|>"
				for idx,iq in enumerate(qs):
					prompt +=  "<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}".format(iq, 'Exact Answer: '+ans_lst[idx])
					if idx<len(qs)-1:
						prompt += '<|eot_id|>'

				# prompt = "<s>[INST] <<SYS>>{}<</SYS>>{} [/INST] Exact Answer:".format("You are a helpful biomedical assistant expert at identifying the core information from answers.  Please respond precisely and concisely.",'Exact Answer:'.join(prompt.split('Exact Answer:')[:-1]))
				

			print(prompt)
			if model_name == 'gpt':
				output = prompt_gpt(prompt)
			else:
				output = prompt_hf(prompt, API_URL ,headers)
			print(output)
			odict[q] = output
			
	
	json.dump(odict, open(outpath,'w'))


def new_extract_orig(inpath, outpath, prompt_name ,col_name,model_name,API_URL,headers,techqa=False):
	with open(inpath,'rb') as f:
		df = pd.read_excel(f)

	prompt_dict = read_prompts()
	prompt_tmp = prompt_dict[prompt_name]
	print(prompt_tmp)

	indices = list(range(df.shape[0]))
	# shuffle(indices)
	# sel_idx = indices[:10]
	sel_idx = indices
	odict = {}
	
	for idx, row in df.iterrows():
		
		if techqa:
			q = row['title'] + '. ' + row['question'].replace('\n','\t')
		else:
			q = row['question'].replace('\n','\t')
		cat = row['question type'].strip()
		if not techqa:
			cat = cat[:-1].lower()
		# print(cat)
		if cat == 'root cause':
			cat = 'resolution'
		if row['remove'] == 'yes':
			print('REMOVE: ', row)
		if idx in sel_idx and row['remove']!='yes':
			sleep(0.5)
			if type(prompt_tmp) == dict:
				single_prompt = prompt_tmp[cat]
			else:
				single_prompt = prompt_tmp

			prompt = single_prompt.replace('{{ques}}',q).replace('{{ans}}',row['ideal answer'].replace('\n','\t'))
			# if 'llama' in outpath:
			# 	prompt = "<s>[INST] {} [/INST]".format('Exact Answer:'.join(prompt.split('Exact Answer:')[:-1]))+'Exact Answer:'
			# print(prompt, cat)
			# print('---'*20)
			examples = prompt.split('\n===\nQuestion:')
			all_qs = []
			all_ans = []
			# print(examples)
			print('---'*20)
			for ex in examples:
				# print(ex)
				iq, ans = ex.split('Exact Answer:')
				all_qs.append(iq)
				all_ans.append(ans)
			# print(all_qs)
			# print('---'*20)
			# print(all_ans)
			# print('---'*20)
			if 'llama' in outpath:
				prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful IT assistant expert at identifying the core information from answers.  Please respond precisely and concisely. <|eot_id|>"
			# 	prompt = "<|begin_of_text|>"
				for idx,iq in enumerate(all_qs):
					
					prompt +=  "<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}".format(iq, 'Exact Answer: '+all_ans[idx])
					if idx<len(all_qs)-1:
						prompt += '<|eot_id|>'
			print(prompt)
			if model_name == 'gpt':
				output = prompt_gpt(prompt)
			else:
				output = prompt_hf(prompt, API_URL, headers)
			print("__"*20,"Output: ",output)
			odict[q] = output
		
			
	olst = []
	for idx,row in df.iterrows():
		
		if techqa:
			q = row['title'] + '. ' + row['question'].replace('\n','\t')
		else:
			q = row['question'].replace('\n','\t')

		if idx in sel_idx and row['remove']!='yes':
			lst = []
			if techqa:
				lst.append(row['title'])
			lst += [row['question'], odict[q], row['ideal answer'], row['question type']]
			olst.append(lst)
	df = pd.DataFrame(olst)
	cols = []
	if techqa:
		cols.append('title')
	cols += ['question', col_name, 'ideal answer', 'question type']
	df.columns = cols
	df.to_excel(outpath, index=False)

def new_extract(inpath, prev_outpath, outpath, prompt_name ,col_name,techqa=False):
	with open(inpath,'rb') as f:
		df = pd.read_excel(f)

	prompt_dict = read_prompts()
	prompt_tmp = prompt_dict[prompt_name]
	print(prompt_tmp)

	indices = list(range(df.shape[0]))
	# shuffle(indices)
	# sel_idx = indices[:10]
	# sel_idx = indices
	odict = {}

	df = pd.read_excel(prev_outpath)

	sel_idx = []

	prev_done = {}

	for idx,row in df.iterrows():
		if techqa:
			q = row['title'] + '. ' + row['question'].replace('\n','\t')
		else:
			q = row['question'].replace('\n','\t')

		if row['pred exact ans'] in ['<Response [422]>','<Response [429]>']:
			sel_idx.append(idx)
		else:
			prev_done[q] = row['pred exact ans']

	for idx, row in df.iterrows():
		
		if techqa:
			q = row['title'] + '. ' + row['question'].replace('\n','\t')
		else:
			q = row['question'].replace('\n','\t')
		cat = row['question type'].strip()
		if not techqa:
			cat = cat[:-1].lower()
		print(cat)
		if cat == 'root cause':
			cat = 'resolution'
		if idx in sel_idx:
			sleep(5)
			if type(prompt_tmp) == dict:
				single_prompt = prompt_tmp[cat]
			else:
				single_prompt = prompt_tmp

			prompt = single_prompt.replace('{{ques}}',q).replace('{{ans}}',row['ideal answer'].replace('\n','\t'))
			if 'llama' in outpath:
				prompt = "<s>[INST] {} [/INST]".format('Exact Answer:'.join(prompt.split('Exact Answer:')[:-1]))+'Exact Answer:'
			print(prompt, cat)
			# output = prompt_gpt(prompt)
			output = prompt_hf(prompt)
			print(output)
			odict[q] = output
		else:
			odict[q] = prev_done[q]
			
	olst = []
	for idx,row in df.iterrows():
		
		if techqa:
			q = row['title'] + '. ' + row['question'].replace('\n','\t')
		else:
			q = row['question'].replace('\n','\t')

		# if idx in sel_idx:
		lst = []
		if techqa:
			lst.append(row['title'])
		lst += [row['question'], odict[q], row['ideal answer'], row['question type']]
		olst.append(lst)
	df = pd.DataFrame(olst)
	cols = []
	if techqa:
		cols.append('title')
	cols += ['question', col_name, 'ideal answer', 'question type']
	df.columns = cols
	df.to_excel(outpath, index=False)



if __name__ == '__main__':
	# ,'../data/BioASQ-training11b/training11b.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json' ../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json', '../data/BioASQ-training11b/Task11BGoldenEnriched/11B3_golden.json'
	# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"
	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
	


	headers = {"Authorization": AUTH_KEY,
		"Content-Type": "application/json"}
# '../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json', '../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json',
	# paths = {'bioasq':['../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json', '../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json']}
	paths = {'bioasq':['../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json', '../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B3_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B4_golden.json']}

	

	# odir = '../results/extraction/llama70b_single_'
	# techqa_path = '../data/techqa_dev_bhavya_0615.xlsx'
	# techqa_ans_path = '../data/techqa_training_and_dev/dev_Q_A.json'
	
	# idir = '../results/extraction/msqa/gpt4_single'
	# odir = '../results/extraction/msqa/gpt4_cat'
	# extract_techqa()
	# main()
	# msqa_paths = ['/Users/bhavya/Documents/qtype_qa/data/msqa/msqa_labeled/msqa_paulina/paulina.json']
	# '/Users/bhavya/Documents/qtype_qa/data/msqa/msqa_labeled/msqa_pilot2/bhavya.json','/Users/bhavya/Documents/qtype_qa/data/msqa/msqa_labeled/msqa_pilot/bhavya.json','/Users/bhavya/Documents/qtype_qa/data/msqa/msqa_labeled/msqa_bhavya/bhavya.json']
	# extract_msqa()


	# msqa_mapping = '../results/extraction/msqa/msqa_changed.json'
	# msqa_labeled_path = '../results/extraction/msqa/final_msqa_labeled.xlsx'
	# msqa_out_path = '../results/extraction/msqa/msqa_changed_gpt_single.xlsx'
	# extract_msqa2()

	# techqa_mapping = '../results/extraction/techqa/techqa_changed.json'
	# techqa_labeled_path = '../results/extraction/techqa/final_techqa_labeled.xlsx'
	# techqa_out_path = '../results/extraction/techqa/techqa_changed_gpt.xlsx'
	# extract_techqa2()

	# write_prompts()


	# techqa_labeled_path = '../results/extraction/techqa/final_techqa_labeled.xlsx'
	# techqa_complete_out_path = '../results/extraction/techqa/techqa_complete_ans_gpt4.xlsx'
	# new_extract(techqa_labeled_path, techqa_complete_out_path, prompt_dict['prompt_tmp_techqa_complete'], True)

	# msqa_labeled_path = '../results/extraction/msqa/final_msqa_labeled.xlsx'
	# msqa_complete_out_path = '../results/extraction/msqa/msqa_complete_ans_gpt4.xlsx'
	# new_extract(msqa_labeled_path, msqa_complete_out_path, 'prompt_tmp_msqa_complete', 'pred complete ans')

	# bioasq_out_path = '../results/extraction/bioasq_complete_ans_gpt4.xlsx'

	# complete_answer_bioasq(paths, bioasq_out_path, bioasq_pred_path)

	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
	# techqa_out_path = '../results/extraction/techqa/llama3_70_cat.xlsx'
	# new_extract_orig(techqa_labeled_path, techqa_out_path, 'cat_prompt_tmp_techqa', 'pred exact ans', True)

	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"

	# msqa_out_path = '../results/extraction/msqa/gpt4_single_sys.xlsx'
	# msqa_out_path2 = '../results/extraction/msqa/llama70_cat_sug2.xlsx'

	# techqa_out_path = '../results/extraction/techqa/llama13_cat_sug.xlsx'
	# techqa_out_path2 = '../results/extraction/techqa/llama13_single_sug2.xlsx'

	

	# new_extract(techqa_labeled_path, techqa_out_path, techqa_out_path2, 'cat_prompt_tmp_techqa', 'pred exact ans', True)

	# new_extract_orig(msqa_labeled_path, msqa_out_path, 'prompt_tmp_msqa', 'pred exact ans', False)

	# msqa_out_path = '../results/extraction/msqa/llama13_cat2.xlsx'
	# new_extract_orig(msqa_labeled_path, msqa_out_path, 'cat_prompt_tmp_msqa2', 'pred exact ans', False)

	# techqa_out_path = '../results/extraction/techqa/gpt4_single_sys.xlsx'
	# new_extract_orig(techqa_labeled_path, techqa_out_path, 'prompt_tmp_techqa', 'pred exact ans', True)
#
	# techqa_out_path = '../results/extraction/techqa/llama13_cat2.xlsx'
	# new_extract_orig(techqa_labeled_path, techqa_out_path, 'cat_prompt_tmp_techqa2', 'pred exact ans', True)


	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"

	# msqa_out_path = '../results/extraction/msqa/llama3_70_single_sys_true.xlsx'

	# new_extract_orig(msqa_labeled_path, msqa_out_path, 'prompt_tmp_msqa', 'pred exact ans', False)

	# msqa_out_path = '../results/extraction/msqa/llama70_cat2.xlsx'
	# new_extract_orig(msqa_labeled_path, msqa_out_path, 'cat_prompt_tmp_msqa2', 'pred exact ans', False)

	# techqa_out_path = '../results/extraction/techqa/llama3_70_single_sys_true.xlsx'
	# new_extract_orig(techqa_labeled_path, techqa_out_path, 'prompt_tmp_techqa', 'pred exact ans', True)

	# techqa_out_path = '../results/extraction/techqa/llama70_cat2.xlsx'
	# new_extract_orig(techqa_labeled_path, techqa_out_path, 'cat_prompt_tmp_techqa2', 'pred exact ans', True)

	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

	# bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B1_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B2_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B3_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B4_golden.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx])

	# bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B1_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B2_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B3_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-8b_11B4_golden_sys.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx], True)


	# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"

	# bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B1_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B2_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B3_golden.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B4_golden.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx])

	# bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B1_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B2_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B3_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B4_golden_sys.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx], True)


	# bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B1_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B2_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B3_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B4_golden_sys.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx], True)

	API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"

	bioasq_pred_path = ['/Users/bhavya/Documents/qtype_qa/results/extraction/llama13b_sys_11B1_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama13b_sys_11B2_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama13b_sys_11B3_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama13b_sys_11B4_golden_sys.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	extract_bioasq(path, bioasq_pred_path[idx], True)

	API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
# 
	# bioasq_pred_path = [ '/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_sys_11B1_golden_sys.json','/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_sys_11B2_golden_sys.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_sys_11B3_golden_sys.json','/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_sys_11B4_golden_sys.json']

	# for idx, path in enumerate(paths['bioasq']):
	# 	if idx == 3:
	# 		extract_bioasq(path, bioasq_pred_path[idx], True)

	bioasq_pred_path = [ '/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_11B1_golden_2.json','/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_11B2_golden_2.json', '/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_11B3_golden_2.json','/Users/bhavya/Documents/qtype_qa/results/extraction/llama70b_11B4_golden_2.json']

	for idx, path in enumerate(paths['bioasq']):
		if idx > 2:
			extract_bioasq(path, bioasq_pred_path[idx], True)






