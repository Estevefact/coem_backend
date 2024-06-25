import nltk
from transformers import AutoTokenizer

nltk.download('punkt')
import json
import os
import re
import timeit

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.cuda.empty_cache() 

class Translator():
  def __init__(self,model="google/flan-t5-xl",tokenizer="google/flan-t5-xl",language_to_translate="english",language_fountain="spanish"):
    if model == "facebook/nllb-200-distilled-600M":
        
        self.tokenizer = NllbTokenizer.from_pretrained(model,src_lang="spa_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        print("Facebook")
    else:
        
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")#, device_map="auto", torch_dtype=torch.float16
        self.tokenizer=T5Tokenizer.from_pretrained(tokenizer)
        print("Google")
    self.short_coem = pd.read_excel("./../coem_seva_cuentos_short_7.xlsx")
    self.language_to_translate=language_to_translate
    self.language_fountain=language_fountain
    self.model.eval()

  def translate(self,text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text,language=self.language_fountain)
    max_chunk_size=496
    translated_texts = []
    for sentence in sentences:
      input_text = f"{sentence}"
      if len(self.tokenizer.encode(input_text, return_tensors='pt')[0]) > max_chunk_size:
        chuncks= sentence.split(", ",maxsplit=2)
        for chunk in chuncks:
          input_text = f"{chunk}"
          chunk_input_ids = self.tokenizer.encode(input_text, return_tensors='pt',max_length=max_chunk_size, truncation=True)#.to("cuda")
          outputs = self.model.generate(chunk_input_ids,forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"], max_new_tokens=max_chunk_size)
          translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
          translated_texts.append(translated_text)
      else:
        chunk_input_ids = self.tokenizer.encode(input_text, return_tensors='pt',max_length=max_chunk_size, truncation=True)#.to("cuda")
        outputs = self.model.generate(chunk_input_ids, max_new_tokens=max_chunk_size, forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"])
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_texts.append(translated_text)
    return " ".join(translated_texts)

  def save(self,text,uuid_story,title,author):
    words = len(re.findall(r'\b\w+\b', text))
    characters = len(text)
    reading_time_minutes = words / 200
    data={"metadata":{"length":characters,"words":words,"reading_time_min":reading_time_minutes},"text":text,"title":title,"author":author}
    try:
      with open(f'./../Cuentos_english/{uuid_story}.json', 'w') as f:
          json.dump(data, f)
    except Exception as e:
      print("Error",e)

  def get_stories_uuids(self, path='./../coem_seva_cuentos_short_7.xlsx'):
    self.short_coem = pd.read_excel(path)
    self.short_coem['Author_name'] = self.short_coem['Name'].fillna('') + ' ' + self.short_coem['LastName'].fillna('')
    return self.short_coem["uuid_story"].tolist()

  def get_df(self,path="./../coem_seva_cuentos_short_7.xlsx"):
    self.short_coem = pd.read_excel(path)
    return self.short_coem

  def get_story(self,uuid_story,df,batch_size=1):
    import json
    texts=[]
    uuids=[]
    index_extract=df[df["uuid_story"]==uuid].index[0]
    title=df.loc[index_extract]["story_name"]
    Name=df.loc[index_extract]["Name"]
    Lastname=df.loc[index_extract]["LastName"]
    try:
      with open(f'./../Cuentos/{uuid_story}.json', 'r') as f:
          data = json.load(f)
          text = data['text']
          uuids.append(uuid_story)
          texts.append(text)
    except Exception as e:
      try:
        with open(f'./../Minicuentos/{uuid_story}.json', 'r') as f:
          data = json.load(f)
          text = data['text']
      except Exception as e:
        print("File not found",e,index_extract)
        text=None
      uuids.append(uuid_story)
      texts.append(text)
    print(len(texts[0]))
    author=Name+" "+Lastname
    return texts[0],uuids[0],author,title

#model="google/flan-t5-xl" #GPU 6 GB
#pip install transformers==4.41.2 nltk

#model="google/flan-t5-xl"

model="facebook/nllb-200-distilled-600M"
T2=Translator(model=model,tokenizer=model)
uuids=T2.get_stories_uuids()
df=T2.get_df()
#To test with one
start = timeit.default_timer()
uuid="3e2c981a-d692-4046-973d-02f4265c3aec"
text,uuid_story,author,title=T2.get_story(uuid,df)
book = T2.translate(text)
T2.save(book,uuid_story,title,author)
took = (timeit.default_timer() - start) * 1.0
print('Code block' + f"story {title} " + ' took: ' + str(took) + ' s')

