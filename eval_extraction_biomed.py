from utils import * 
from rouge_score import rouge_scorer
import pandas as pd 
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
sample_qs = ['Does silencing of SRRM4 inhibit tumor growth across cancers?','What is the life expectancy for Duchenne muscular dystrophy patients?','What processes do orexin/hypocretin neurons regulate?']
remove_words = '''
Identify the exact answer from the ideal answer for the given question below. If the exact answer is a list, separate each entry by semi-colon.
Question: Does silencing of SRRM4 inhibit tumor growth across cancers?
Ideal Answer: No, silencing of SRRM4 promotes tumor growth across cancers.
Exact Answer: no
====
Question: What is the life expectancy for Duchenne muscular dystrophy patients?
Ideal Answer: The life expectancy for Duchenne muscular dystrophy patients is around 28.1 years.
Exact Answer: 28.1 years of age
====
Question: What processes do orexin/hypocretin neurons regulate?
Ideal Answer: The hypocretin/orexin neurons play a role in regulating wakefulness/sleep cycles, pain perception, and appetite. Orexin/hypocretin neurons regulate multiple homeostatic and behavioral processes, including arousal, metabolism, and reward. Orexin/hypocretin neurons regulate a wide range of physiological functions, including sleep/wakefulness state regulation, feeding behavior and energy homeostasis, reward system, and aging and neurodegenerative diseases. Orexin/hypocretin neurons regulate wakefulness/sleep cycles, pain perception, appetite, feeding behavior, energy homeostasis, goal-directed behaviors, and the reward system. They are also involved in the diagnosis and symptoms of narcolepsy type 1, and can be targeted by antagonists to promote sleep initiation and maintenance. Orexin/hypocretin neurons regulate a variety of processes, including wakefulness, appetite, reward-seeking behavior, stress response, and energy homeostasis. They also play a role in regulating sleep-wake cycles, learning and memory, and autonomic functions such as blood pressure and heart rate. Orexin/hypocretin neurons regulate a variety of processes, including wakefulness, arousal, reward, motivation, feeding, and stress. Orexin/hypocretin neurons regulate wakefulness/sleep cycles, pain perception, appetite, food intake, reward system, and energy homeostasis. They also mediate goal-directed behaviors. Antagonists that target both orexin-1 and orexin-2 receptors have been shown to promote sleep initiation and maintenance. What processes do orexinhypocretin neurons regulate Orexin/hypocretin neurons regulate feeding behavior, sleep/wakefulness states, and goal-directed behaviors.
Exact Answer: sleep; appetite; wakefullness; pain; energy homeostasis; goal-directed behaviors; arousal; addiction
====
Question: {{ques}}
Ideal Answer: {{ans}}
Exact Answer: '''
def calc_rouge(targets, pred):
	return scorer.score_multi(targets, pred)['rougeL']
def calc_len(txts):
	words_len = []
	for t in txts:
		# print(t.split())
		# print(t)
		words_len.append(len(t.split()))
	df = pd.DataFrame(words_len)

	print(df.describe())

def calc_res(paths, pred_prefix, suffix):
	cat_dict = {'all':[]}
	cat_dict_len_qs = {'all':[]}
	cat_dict_len_idans = {'all':[]}
	cat_dict_len_exans = {'all':[]}
	for path in paths['bioasq']:
		# pred_path = pred_prefix+path.split('/')[-1].replace('.json','')+'_2.json'
		pred_path = pred_prefix+path.split('/')[-1].replace('.json','')+suffix
		print(pred_path)
		pred_dict = read_pred(pred_path)
		bioasq_qs, bioasq_cats, bioasq_idans, bioasq_exans = read_bioasq(path)
		
		for idx, c in enumerate(bioasq_cats):
			if len(bioasq_exans[idx])>0:
				id_ans = ''
				for a in bioasq_idans[idx]:
						id_ans += a.strip()+' '
				# tmp = remove_words.replace('{{ques}}',bioasq_qs[idx]).replace('{{ans}}',id_ans)
				# print(pred_dict[bioasq_qs[idx]])
				# pred = pred_dict[bioasq_qs[idx]].split('Exact Answer:')[-1].strip()
				try:
					pred = pred_dict[bioasq_qs[idx]].split('Exact Answer:')[4].strip().split('assistant')[0]
				except:
					pred = pred_dict[bioasq_qs[idx]].split('Exact Answer:')[-1].strip().split('assistant')[0]
				# print('----',pred)
				# if '*' in pred:
				# 	pred = pred.split('*')[-1]
				# elif 'Question 4' in pred:
				# 	pred = pred.split('Question 4:')[1]
				
				# print('----',pred)
				# print([' '.join(ex) for ex in bioasq_exans[idx]] , bioasq_exans[idx], bioasq_cats[idx])
				# print(bioasq_idans[idx])
				try:
					cat_dict[c].append((bioasq_exans[idx], pred))
					cat_dict_len_qs[c].append(bioasq_qs[idx])
					cat_dict_len_idans[c] += bioasq_idans[idx]
					cat_dict_len_exans[c] += [' '.join(ex) for ex in bioasq_exans[idx]] 
				except:
					cat_dict[c] = [(bioasq_exans[idx], pred)]
					cat_dict_len_qs[c] = [bioasq_qs[idx]]
					cat_dict_len_idans[c] =  bioasq_idans[idx]
					cat_dict_len_exans[c] = [' '.join(ex) for ex in bioasq_exans[idx]] 
				cat_dict['all'].append((bioasq_exans[idx], pred))
				cat_dict_len_qs['all'].append(bioasq_qs[idx])
				# print(bioasq_idans[idx], cat_dict_len_idans['all'])
				cat_dict_len_idans['all'] += bioasq_idans[idx]
				cat_dict_len_exans['all']+= [' '.join(ex) for ex in bioasq_exans[idx]] 
		
	for k,v in cat_dict_len_qs.items():
		print(k,'Lens Questions')
		calc_len(v)

	for k,v in cat_dict_len_idans.items():
		print(k,'Lens Id Ans')
		calc_len(v)

	for k,v in cat_dict_len_exans.items():
		print(k,'Lens Exact')
		calc_len(v)

	out_dict = {}

	for k,vs in cat_dict.items():
		rgp = 0
		rgr = 0
		rgf = 0
		for v in vs:
			ref = v[0]
			hyp = []
			for h in v[1].split(';'):
				hyp.append(h)
			inp = 0
			inr = 0
			inf = 0
			# print(1, ref, hyp)
			for r in ref:
		
				bf = 0
				brscore = (0.0, 0.0,0.0)
				
				for h in hyp:
					

					rscore = calc_rouge(r, h)
					
					if rscore[2] > bf:
						brscore = rscore
					bf = max(rscore[2], bf)

				inp += brscore[0]
				inr += brscore[1]
				inf += brscore[2]
			# print(inp, inp/len(ref))
			rgp += inp/len(ref)
			rgr += inr/len(ref)
			rgf += inf/len(ref)

		out_dict[k] = {'prec1': rgp/len(vs)*100, 'rec1': rgr/len(vs)*100, 'f11':rgf/len(vs)*100}
		print(k, len(vs),round(rgp/len(vs)*100,2)  , round(rgr/len(vs)*100,2), round(rgf/len(vs)*100,2))


	for k,vs in cat_dict.items():
		rgp = 0
		rgr = 0
		rgf = 0
		for v in vs:
			ref = v[0]
			hyp = []

			for h in v[1].split(';'):
				hyp.append(h)
			# print(2, ref, hyp)
			inp = 0
			inr = 0 
			inf = 0
			for h in hyp:
				bf = 0
				brscore = (0.0, 0.0,0.0)

				for r in ref:

					rscore = calc_rouge(r, h)
					
					if rscore[2] > bf:
						brscore = rscore
					bf = max(rscore[2], bf)

				inp += brscore[0]
				inr += brscore[1]
				inf += brscore[2]
			# print(inp, inp/len(ref))
			rgp += inp/len(hyp)
			rgr += inr/len(hyp)
			rgf += inf/len(hyp)
		out_dict[k]['prec2']= rgp/len(vs)*100
		out_dict[k]['rec2']= rgr/len(vs)*100
		out_dict[k]['f12'] = rgf/len(vs)*100
		print(k, len(vs),round(rgp/len(vs)*100,2)  , round(rgr/len(vs)*100,2), round(rgf/len(vs)*100,2))

	return out_dict

	

if __name__ == '__main__':
	
	paths = {'bioasq':['../data/BioASQ-training11b/Task11BGoldenEnriched/11B1_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B2_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B3_golden.json','../data/BioASQ-training11b/Task11BGoldenEnriched/11B4_golden.json']}
	pred_prefix = '../results/extraction/llama70b_'
	calc_res()