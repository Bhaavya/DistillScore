from core_key_extractor_gpt import * 
from eval_extraction_biomed import * 
import numpy as np 



def calc_stats(prefix_path):
	all_res_dict = {}
	for i in range(3):
		if i == 0:
			suffix = '_sys.json'
		else:
			suffix = '_sys_{}.json'.format(i-1)
		print(suffix)

		res_dict = calc_res(paths, prefix_path, suffix)
		for k,vs in res_dict.items():
			try:
				all_res_dict[k]
			except:
				all_res_dict[k] = {'prec1':[],'rec1':[],'f11':[], 'prec2':[],'rec2':[],'f12':[]}

			for k2,v2 in vs.items():
				all_res_dict[k][k2].append(v2)
	print(all_res_dict)
	for k1,v1 in all_res_dict.items():
		print(k1)
		for k2,v2 in v1.items():
			print(k2, round(np.average(np.array(v2)),3),'±',round(np.std(np.array(v2)),3))


if __name__ == '__main__':
	AUTH_KEY = AUTH_KEY
	headers = {"Authorization": AUTH_KEY,
		"Content-Type": "application/json"}

	API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"

	paths = {'bioasq':['../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json', '../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B3_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B4_golden.json']}

	# for i in range(2):
	# 	if i>0:
	# 		bioasq_pred_path = [ '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B1_golden_sys_{}.json'.format(i),'/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B2_golden_sys_{}.json'.format(i), '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B3_golden_sys_{}.json'.format(i),'/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_11B4_golden_sys_{}.json'.format(i)]

	# 		for idx, path in enumerate(paths['bioasq']):
	# 			extract_bioasq(path, bioasq_pred_path[idx],'llama',API_URL, headers,  True)


	# 	bioasq_pred_path = [ '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B1_golden_sys_{}.json'.format(i),'/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B2_golden_sys_{}.json'.format(i), '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B3_golden_sys_{}.json'.format(i),'/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_11B4_golden_sys_{}.json'.format(i)]

	# 	for idx, path in enumerate(paths['bioasq']):
	# 		extract_bioasq(path, bioasq_pred_path[idx],'gpt',API_URL, headers,  True)

	prefix_path = '/Users/bhavya/Documents/qtype_qa/results/extraction/llama3-70b_'

	calc_stats(prefix_path)

	prefix_path = '/Users/bhavya/Documents/qtype_qa/results/extraction/gpt4_sys_'

	calc_stats(prefix_path)