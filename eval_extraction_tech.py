from rouge_score import rouge_scorer
import pandas as pd 

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calc_len(txts):
	words_len = []
	for t in txts:
		# print(t.split())
		# print(t)
		words_len.append(len(t.split()))
	df = pd.DataFrame(words_len)

	print(df.describe())

def calc_rouge(targets, pred):
	return scorer.score_multi(targets, pred)['rougeL']

def main_calc(path):
	df = pd.read_excel(path) 
	cat_dict = {'all':[]}
	cat_dict_len_qs = {'all':[]}
	cat_dict_len_idans = {'all':[]}
	cat_dict_len_exans = {'all':[]}
	label_dict = {}

	for idx, row in df.iterrows():
		if type(row['true exact ans'])!=float:
			
			label_dict[row['question']] = [row['true exact ans']]
			if type(row['pred exact cat ans']) == float:
				v0 = ''
			else:
				v0 = row['pred exact cat ans'].split('Exact Answer:')[-1].strip()

			if type(row['pred exact ans']) == float:
				v1 = ''
			else:
				v1 = row['pred exact ans'].split('Exact Answer:')[-1].strip()
			cat_dict['all'].append([[row['true exact ans']], v0 , v1])
			cat = row['question type'].strip()
			if 'msqa' in path:
				cat = cat.lower()
			if cat == 'Error resolution - cause':
				cat = 'Error resolution'
			try:
				cat_dict[cat].append([[row['true exact ans']], v0, v1])
				cat_dict_len_qs[cat].append(row['question'])
				cat_dict_len_idans[cat].append(row['ideal answer'])
				cat_dict_len_exans[cat].append(row['true exact ans'])
			except:
				cat_dict[cat] = [[[row['true exact ans']], v0, v1]]
				cat_dict_len_qs[cat] = [row['question']]
				cat_dict_len_idans[cat] = [row['ideal answer']]
				cat_dict_len_exans[cat] = [row['true exact ans']]


			cat_dict_len_qs['all'].append(row['question'])
			# print(bioasq_idans[idx], cat_dict_len_idans['all'])
			cat_dict_len_idans['all'].append(row['ideal answer'])
			cat_dict_len_exans['all'].append(row['true exact ans'])

	for k,v in cat_dict_len_qs.items():
		print(k,'Lens Questions')
		calc_len(v)

	for k,v in cat_dict_len_idans.items():
		print(k,'Lens Id Ans')
		calc_len(v)

	for k,v in cat_dict_len_exans.items():
		print(k,'Lens Exact')
		calc_len(v)

	miss_cnt = 0
	print(cat_dict.keys())
	for k,vs in cat_dict.items():

		rgpc = 0
		rgrc = 0
		rgfc = 0

		rgps = 0
		rgrs = 0
		rgfs = 0

		cnt = 0

		for v in vs:
			p, r, f = calc_rouge(v[0], v[1])
			rgpc += p
			rgrc += r
			rgfc += f 
			if v[2] == '':
				miss_cnt += 1
				p, r, f = 0, 0, 0
			else:
				p, r, f = calc_rouge(v[0], v[2])

			
			rgps += p
			rgrs += r
			rgfs += f 

			cnt += 1
		print('GPT 3.5')
		print(k,cnt)
		print('Categorical prompt')
		print(rgpc/float(cnt))
		print(rgrc/float(cnt))
		print(rgfc/float(cnt))
		print('Single prompt')
		print(rgps/float(cnt))
		print(rgrs/float(cnt))
		print(rgfs/float(cnt))
		print(miss_cnt)
	return label_dict


def other_calc(path, label_dict):
	df = pd.read_excel(path) 
	cat_dict = {'all':[]}
	det_out = []

	for idx, row in df.iterrows():
		try:
			label_dict[row['question']]
		except Exception as e:
		
			continue
	
		if type(row['pred exact ans']) == float:
				v0 = ''
		else:	
				
				idx_ex = 4
				# if 'cat' in path:
				# 	idx_ex = 2
				try:
					v0 = row['pred exact ans'].split('Exact Answer:')[idx_ex].strip().split('assistant')[0]
				except:
					v0 = row['pred exact ans'].split('Exact Answer:')[-1].strip().split('assistant')[0]
				 
		# print(v0, label_dict[row['question']])

		cat_dict['all'].append([label_dict[row['question']], v0, row['ideal answer'], row['question']])
		
		cat = row['question type'].strip()
		if 'msqa' in path:
			cat = cat.lower()
		if cat == 'Error resolution - cause':
			cat = 'Error resolution'
		try:
			cat_dict[cat].append([label_dict[row['question']], v0, row['ideal answer'], row['question']])
		except:
			cat_dict[cat] = [[label_dict[row['question']], v0, row['ideal answer'], row['question']]]

	out_dict = {}

	for k,vs in cat_dict.items():

		rgp = 0
		rgr = 0
		rgf = 0

		cnt = 0

		for v in vs:
			
			p, r, f = calc_rouge(v[0], v[1])
			det_out.append([v[0],v[1],v[2],v[3],k,p,r,f])
			rgp += p
			rgr += r
			rgf += f 
			cnt += 1

		print(path)
		print(k, cnt)
		print(rgp/float(cnt))
		print(rgr/float(cnt))
		print(rgf/float(cnt))
		out_dict[k] = {'prec': rgp/float(cnt), 'rec': rgr/float(cnt), 'f1': rgf/float(cnt)}
	det_df = pd.DataFrame(det_out)
	det_df.to_excel('../results/extraction/msqa/llama3_70_single_sys_true_det_out.xlsx',index=False)
	return out_dict
	

if __name__ == '__main__':
	
	
	# main_path = '../results/extraction/techqa/final_techqa_labeled.xlsx'
	main_path = '../results/extraction/msqa/final_msqa_labeled.xlsx'
	# paths = ['../results/extraction/msqa/cat_mixtral.xlsx', '../results/extraction/msqa/mixtral_single.xlsx', '../results/extraction/msqa/llama13b_single.xlsx', '../results/extraction/msqa/cat_llama13b.xlsx', '../results/extraction/msqa/cat_llama70b.xlsx', '../results/extraction/msqa/llama70b_single.xlsx', '../results/extraction/msqa/gpt3.5_single.xlsx', '../results/extraction/msqa/gpt3.5_cat.xlsx','../results/extraction/msqa/gpt4_single.xlsx', '../results/extraction/msqa/gpt4_cat.xlsx']
	
	# paths =['../results/extraction/techqa/llama3_70_single.xlsx', '../results/extraction/techqa/llama3_70_cat.xlsx','../results/extraction/techqa/llama3_8_single.xlsx','../results/extraction/techqa/llama3_8_cat.xlsx','../results/extraction/techqa/llama3_70_single_sys.xlsx', '../results/extraction/techqa/llama3_70_cat_sys.xlsx','../results/extraction/techqa/llama3_8_single_sys.xlsx','../results/extraction/techqa/llama3_8_cat_sys.xlsx']
	paths =['../results/extraction/msqa/llama3_70_single_sys_true.xlsx']
	 # '../results/extraction/techqa/llama3_8_single_sys_true.xlsx']
	# paths =['../results/extraction/techqa/llama3_8_cat2.xlsx', '../results/extraction/techqa/llama3_70_cat2.xlsx','../results/extraction/techqa/llama13_cat2.xlsx','../results/extraction/techqa/llama70_cat2.xlsx']
	# paths =['../results/extraction/techqa/llama3_70_single_sys_true.xlsx', '../results/extraction/techqa/llama3_8_single_sys_true.xlsx']
	label_dict = main_calc(main_path)
	for path in paths:
		other_calc(path, label_dict)


