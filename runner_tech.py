from core_key_extractor_gpt import * 
from eval_extraction_tech import * 
import numpy as np 

def calc_stats(root_path, main_path):

	label_dict = main_calc(main_path)
	all_res_dict = {}
	for i in range(3):
		if i == 0:
			path = root_path +'.xlsx'
		else:
			path = root_path + '_{}.xlsx'.format(i-1)

		res_dict = other_calc(path, label_dict)
		for k,vs in res_dict.items():
			try:
				all_res_dict[k]
			except:
				all_res_dict[k] = {'prec':[],'rec':[],'f1':[]}

			for k2,v2 in vs.items():
				all_res_dict[k][k2].append(v2)

	for k1,v1 in all_res_dict.items():
		print(k1)
		for k2,v2 in v1.items():
			print(k2, round(np.average(np.array(v2)),3),'±',round(np.std(np.array(v2)),3))



if __name__ == '__main__':
	headers = {"Authorization": AUTH_KEY,
		"Content-Type": "application/json"}
	techqa_labeled_path = '../results/extraction/techqa/final_techqa_labeled.xlsx'
	msqa_labeled_path = '../results/extraction/msqa/final_msqa_labeled.xlsx'

	API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"

	# for i in range(2):
	# 	if i >0:
	# 		techqa_out_path = '../results/extraction/techqa/gpt4_single_sys_{}.xlsx'.format(i)
	# 		new_extract_orig(techqa_labeled_path, techqa_out_path, 'prompt_tmp_techqa', 'pred exact ans', 'gpt' ,'', '',True)

	# 		techqa_out_path = '../results/extraction/techqa/llama3_70_single_sys_true_{}.xlsx'.format(i)
	# 		new_extract_orig(techqa_labeled_path, techqa_out_path, 'prompt_tmp_techqa', 'pred exact ans', 'llama' ,API_URL,headers, True)

	# 		msqa_out_path = '../results/extraction/msqa/gpt4_single_sys_{}.xlsx'.format(i)
	# 		new_extract_orig(msqa_labeled_path, msqa_out_path, 'prompt_tmp_msqa', 'pred exact ans', 'gpt' ,'','', False)

	# 	msqa_out_path = '../results/extraction/msqa/llama3_70_single_sys_true_{}.xlsx'.format(i)
	# 	new_extract_orig(msqa_labeled_path, msqa_out_path, 'prompt_tmp_msqa', 'pred exact ans', 'llama' , API_URL, headers, False)


	root_path = '../results/extraction/techqa/gpt4_single_sys'
	main_path = '../results/extraction/techqa/final_techqa_labeled.xlsx'
	calc_stats(root_path, main_path)

	root_path = '../results/extraction/techqa/llama3_70_single_sys_true'
	calc_stats(root_path, main_path)

	main_path = '../results/extraction/msqa/final_msqa_labeled.xlsx'
	root_path = '../results/extraction/msqa/gpt4_single_sys'
	calc_stats(root_path, main_path)

	root_path = '../results/extraction/msqa/llama3_70_single_sys_true'
	calc_stats(root_path, main_path)


	

