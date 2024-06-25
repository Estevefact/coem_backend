import json
import os

import nltk
import pandas as pd
import torch
from TTS.api import TTS

nltk.download('punkt')


#torch.cuda.empty_cache() 

class Voiceself():
    def __init__(self,language="spanish"):
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # List available 🐸TTS models
        self.available_models = TTS().list_models()
        # Init TTS with a specific model
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)#, gpu=True
        self.tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
        self.language=language
        self.short_coem = pd.read_excel("./../coem_seva_cuentos_short_7.xlsx")
        self.max_length=196
        self.tts.compile()
        self.tts.eval()


    def save_model_locally(self,path=',/saved_tts_model'):
      # this needs to have the config file not null in order to be retrieved later
      if not os.path.exists(path):
        os.makedirs(path)
      model_save_path = path
      config_save_path = model_save_path + '/config.json'
      vocoder_save_path = model_save_path + '/vocoder.pth'

      # Save the model weights
      torch.save(self.tts.state_dict(), model_save_path + '/model.pth')

      with open(config_save_path, 'w') as f:
        json.dump(self.tts.config, f)


    def load_model_locally(self,path=',/saved_tts_model'):
      if not os.path.exists(path):
        os.makedirs(path)
      model_save_path = path
      config_save_path = model_save_path + '/config.json'
      vocoder_save_path = model_save_path + '/vocoder.pth'

      # Initialize the TTS model with the config
      tts_loaded= TTS(model_path=model_save_path,config_path=config_save_path)

      # Load the model weights
      tts_loaded.model.load_state_dict(torch.load(model_save_path + '/model.pth'))
      tts_loaded.model.compile()
      tts_loaded.model.eval()
      return tts_loaded


    def extract_text(self,index_story,batch_size=1):
      texts=[]
      uuids=[]
      for index_extract in range(index_story,index_story+batch_size):
        uuid_story=self.short_coem.loc[index_extract]["uuid_story"]
        title=self.short_coem.loc[index_extract]["story_name"]
        Name=self.short_coem.loc[index_extract]["Name"]
        Lastname=self.short_coem.loc[index_extract]["LastName"]
        try:
          with open(f'./../Cuentos/{uuid_story}.json', 'r') as f:
              data = json.load(f)
              text = data['text']
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
      return texts[0],uuids[0],title,Name,Lastname

    def split_text_into_sentences(self, text):
        sentences = self.tokenizer.tokenize(text)
        return sentences

    def split_into_chunks(self, sentence):
        """
        Split the text into chunks of a specified maximum length. if the sentence is too big

        :param sentence: The sentence to split
        :return: A list of text chunks
        """
        unfinnished=False
        max_length=self.max_length
        if len(sentence) <= max_length:
            return [sentence],unfinnished
        else:
          unfinnished=True
          words = sentence.split()
          chunks = []
          chunk = ""

          for word in words:
              if len(chunk) + len(word) + 4 <= max_length:
                  chunk = chunk + " " + word if chunk else word
              else:
                  chunk+=","
                  chunks.append(chunk)
                  chunk = word

          if chunk:
              chunks.append(chunk)
          final_chunks=[]
          for chunk in chunks:
            parts = chunk.split(',')
            # Check if there are at least 5 commas
            if len(parts) < 5:
              final_chunks.append(chunk)
                  # Return the original sentence and None
            else:
              # Join the first part up to the fourth comma
              first_sentence = ','.join(parts[:4]) + ','

              # Join the remaining parts to form the second sentence
              second_sentence = ','.join(parts[4:])

              final_chunks.append(first_sentence)
              final_chunks.append(second_sentence)
          return final_chunks,unfinnished
      
    def test_one_sentence_generation(self,language="en",text=""" They talk by flapping their meat at each other""",file_path="sentence-u0_78.wav",speaker_wav = ',/audio_references/media-ls-sagan-1958124-3-1.mp3'):
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)

    def generate_voice(self, text, speaker_wav= './../media-ls-sagan-1958124-3-1.mp3', directory_path=f"./../stories_audios", story_name="21727239273",title="Prueba",Name="Desconocido",Lastname="Desconocido"):
        """
        Generate a voice file from the given text and save it to the specified path.
        This will generate voices in chuncks for one to unite them in order
        :param text: The text to convert to speech
        :param speaker_wav: The path to the reference speaker audio file
        :param language: The language of the text
        :param file_path: The path where the generated voice file will be saved
        """
        
        language="es"
        if self.language=="spanish":
          language="es"
          title= title + ", de "+ Name +" "+ Lastname + "."
          directory_path = os.path.join(directory_path, "spanish")
        elif self.language=="english":
          language="en"
          title=title+ ", by "+ Name +" "+ Lastname + "."
          directory_path = os.path.join(directory_path, "english")
        directory_path = os.path.join(directory_path, story_name)
        if not os.path.exists(directory_path):
          os.makedirs(directory_path)
        sentences = self.split_text_into_sentences(text)
        try:
          with open(f'./current_embeded_number.txt', 'r') as f:
            data = json.load(f)
            counter = data['number']
            i = data['i']
        except Exception as e:
          print("No saved number of file",e)
          i=0# track the amount of audios generated per story
          counter=0# track where the index of sentences is
        file_path = os.path.join(directory_path, "sentence_0.wav")
        self.tts.tts_to_file(text=title, speaker_wav=speaker_wav, language=language, file_path=file_path)
        while counter < len(sentences):
          sentence = sentences[counter]
          chunks,unfinnished = self.split_into_chunks(sentence)
          sentence_index=0
          for j, chunk in enumerate(chunks):
            if unfinnished:
              namewav=f"sentence_{i + 1}.wav"
            else:
              namewav=f"sentence-u{sentence_index}_{i + 1}.wav"
            file_path = os.path.join(directory_path, namewav)
            sentence_index+=1
            if chunk:
              if len(chunk)==0:
                pass
              else:
                #Filter signal inputs
                if chunk == "," or chunk == "?" or chunk == "." or chunk == ";" or chunk == "¡" or chunk == "-" or chunk == "!" or chunk == '"' or chunk == '(' or chunk == ')' or chunk == ':':
                  continue
                #Erase last commas just in case
                if ('"' in chunk[-1]) or ("." in chunk[-1])or (":" in chunk[-1])or (chunk[-1] == "”"):
                  chunk=chunk[:-1]
                #Erase commas at the start
                if ('"' == chunk[0]) or ("." == chunk[0])or (":" == chunk[0]) or (chunk[0] == "”"):
                  chunk=chunk[1:]
                if str(chunk):
                  self.tts.tts_to_file(text=chunk, speaker_wav=speaker_wav, language=language, file_path=file_path)
                i+=1
            else:
              pass
          counter+=1
          current={"number":counter,"i":i}
          if counter%5:
            print(counter)
          with open(f'./current_voice_number.txt', 'w') as f:
                json.dump(current, f)


generator = Voiceself(language="spanish")

indexes = generator.short_coem.index.tolist()
# numbers that didn't work:
indexes=indexes[250::-1]
start=0
try:
    with open(f'./current_story_number.txt', 'w') as f:
            data = json.load(f)
            start=data["number"]
except Exception as E:
    print("No where to start")
indexes = indexes[start:]
uuids_generated=[]
count=start
for storie in indexes:
  extracted_text, uuid_story, title, name, lastname = generator.extract_text(index_story=storie,batch_size=1)
  generator.generate_voice(text=extracted_text,story_name=uuid_story,title=title,Name=name,Lastname=lastname)
  uuids_generated.append(uuid_story)
  print("AAAAAAAAAAAAAAA",uuid_story,count)
  count+=1
  current={"number":count}
  with open(f'./current_story_number.txt', 'w') as f:
        json.dump(current, f)