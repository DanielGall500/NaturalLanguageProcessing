import spacy

nlp = spacy.load('en')

sentence = nlp("Kafka walked over to Baku in Azerbaijan")

for token in sentence:
	print(token.text, token.ent_type_)
print("\n")

for ent in sentence.ents:
	print(ent.text, ent.label_)
print("\n")

book_f = open("corpus/metamorphosis_kafka.txt").read()
parse_book = nlp(book_f)
book_f.close()

for ent in parse_book.ents:
	print(ent.text, ent.label_)



