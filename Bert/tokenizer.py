test_sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
model_name='distilbert-base-uncased-finetuned-sst-2-english'

from transformers import AutoTokenizer,AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_name)
# len(input_ids)=len(attention_mask)
model=AutoModelForSequenceClassification.from_pretrained(model_name)
# what tokenizer does?
batch_input=tokenizer(test_sentences,truncation=True,padding=True,return_tensors='pt')
print(batch_input)