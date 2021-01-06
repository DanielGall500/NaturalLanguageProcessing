import spacy
import plac
import random
from pathlib import Path

TRAIN_DATA = [
	('Who is Kafka?', {
		'entities' : [(7, 12, 'PERSON')]
		}),
	('He was born in Prague and died in Kierling', {
		'entities' : [(15, 21, 'LOC'), (33, 41, 'LOC')]
		})
]

@plac.annotations(
	model = ("NER Model", "option", "m", str),
	output_dir=("Optional output directory", "option", "o", Path),
	n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, output_dir="models/kafka_NER_model", n_iter=100):
	#Load Model
	if model is not None:
		nlp = spacy.load(model)
		print("Loaded Model: %s" % model)
	else:
		nlp = spacy.load('en')
		print("Created Blank 'en' model")

	#Create Pipeline
	if 'ner' not in nlp.pipe_names:
		ner = nlp.create_pipe('ner')
		nlp.add_pipe(ner, last=True)
	else:
		ner = nlp.get_pipe('ner')

	#Add Labels
	for _, annotations in TRAIN_DATA:
		for ent in annotations.get('entities'):
			ner.add_label(ent[2])

	#Get names of other pipes and disable them during training
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

	#Only train NER
	with nlp.disable_pipes(*other_pipes):
		optimizer = nlp.begin_training()

		for itn in range(n_iter):
			random.shuffle(TRAIN_DATA)
			losses = {}

			for text, annotations in TRAIN_DATA:
				#Heavy lifting
				nlp.update(
					[text],
					[annotations],
					drop=0.5,
					sgd=optimizer,
					losses=losses)
			print(losses)

	#Test the trained model
	for text, _ in TRAIN_DATA:
		doc = nlp(text)
		print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
		print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


	#Save model to output directory
	if output_dir is not None:
		output_dir = Path(output_dir)

		if not output_dir.exists():
			output_dir.mkdir()
	nlp.to_disk(output_dir)
	print("Saved model to %s" % output_dir)

	#Test the saved model
	print("Loading from %s" % output_dir)
	nlp_v2 = spacy.load(output_dir)

	for text, _ in TRAIN_DATA:
		doc = nlp_v2(text)
		print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
		print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

if __name__ == '__main__':
	plac.call(main)

















