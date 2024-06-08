import json

def read_bioasq(path):
	with open(path) as f:
		data = json.load(f)
	ques = []
	cats = []
	ans_id = []
	ans_ex = []
	for q in data['questions']:
		ques.append(q['body'])
		cats.append(q["type"])
		ans_id.append(q["ideal_answer"])
		
		try:
			if type(q['exact_answer'])!= list:
				ans_ex.append([[q['exact_answer']]])
				# print(q['exact_answer'],[[q['exact_answer']]],cats[idx])
			else:
				ans_ex.append(q['exact_answer'])
		except:
			ans_ex.append([])
	return ques, cats, ans_id, ans_ex

def read_pred(path):
	with open(path) as f:
		data = json.load(f)
	return data
