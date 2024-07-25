import json
import os
import random

import pandas as pd
from pydub import AudioSegment


class AudioProcessor():
  def _init_(self,language="spanish"):
    self.short_coem = pd.read_excel('./coem_cuentos_authors.xlsx')
    self.short_coem['Author_name'] = self.short_coem['Name'].fillna('') + ' ' + self.short_coem['LastName'].fillna('')
    self.language=language

  def parsersentence(self,test):
    try:
      unnfinnished=test.split("_")[0].split("-")[1]
      if unnfinnished[0]=="u":
        return True
      else:
        return False
    except IndexError as e:
      unnfinnished=None
    return False

  def returnnumberformfile(self,test):
    try:
      unnfinnished=test.split("_")[0].split("-")[1]
      if unnfinnished[0]=="u":
        return int(unnfinnished[1])
      else:
        return 42
    except IndexError as e:
      unnfinnished=None
    return 42

  def sentencenumber(self,test):
    try:
      return int(test.split('_')[1].split('.')[0])
    except Exception as e:
      return 1000

  def filesready(self,path="./stories_audios/"):
    path=path+"spanish"+"/"
    readyfiles=[]
    for filename in os.listdir(path):
      filepath=path+filename
      readyfiles.append(filepath)
    return readyfiles

  def clean_done_wavs(self,path):
    for filename in os.listdir(path):
      filepath=path+filename
      os.remove(filepath)

  def concatenate_audio_files(self,folder_path_start, uuid_story="They_are_made_out_of_meat_terry"):
      combined_audio = AudioSegment.empty()

      # Get a sorted list of all wav files in the folder
      files = sorted([f for f in os.listdir(folder_path_start) if f.endswith('.wav') and f.startswith('sentence')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))

      # Concatenate all audio files
      for filename in files:
          file_path = os.path.join(folder_path_start, filename)
          audio = AudioSegment.from_wav(file_path)
          combined_audio += audio
          sentence_number=self.sentencenumber(filename)
          unnfinnished=self.parsersentence(filename)
          number=self.returnnumberformfile(filename)
          filenext=f"sentence-u{number+1}_{sentence_number}.wav"
          same_sentence_next=False
          if filenext in files:
            same_sentence_next=True
          if not unnfinnished:
            pause_duration = random.randint(500, 700)
            silence = AudioSegment.silent(duration=pause_duration)
            combined_audio += silence
          elif unnfinnished and same_sentence_next:
            pause_duration = random.randint(200, 400)
            silence = AudioSegment.silent(duration=pause_duration)
            combined_audio += silence
          else:
            pause_duration = random.randint(400, 700)
            silence = AudioSegment.silent(duration=pause_duration)
            combined_audio += silence

      lastpath=folder_path_start.split("/")[-1]
      uuid=uuid_story.split("/")[-1]
      uuid_story

      # Save the combined audio file
      folder_path = os.path.join(folder_path_start, f'{lastpath}.wav')

      title=self.short_coem[self.short_coem["uuid_story"]==uuid_story]["story_name"]
      author=self.short_coem[self.short_coem["uuid_story"]==uuid_story]["Author"]

      # Export the combined audio file
      saving_path=f'./missing_audios/{uuid}.mp3'
      combined_audio.export(saving_path, format="mp3", tags={'artist': 'Carl Sagan Coem', 'album': author ,'original_title': title, "language": self.language, 'comments': 'Coem español'})


# Folder containing the audio files
language="spanish"
folder_path = './'
audioprocessor= AudioProcessor()
audioprocessor.language="spanish"
audioprocessor.short_coem = pd.read_excel('./coem_cuentos_authors.xlsx')
filestoprocess=audioprocessor.filesready(folder_path)
totalfiles=len(filestoprocess)
print(totalfiles)
start=0
for uuid_story in filestoprocess:
  folder_path_story=folder_path+uuid_story+"/"
  print(f"Processing {folder_path_story}")
  audioprocessor.concatenate_audio_files(folder_path_start=uuid_story,uuid_story=uuid_story)
  start+=1
  if start%10==0:
    print(f"Processed {start} of {totalfiles}")
  data={"uuid_story":uuid_story,"number":start}
  with open(f'./current{language}.json', 'w') as f:
    metadata = json.dump(data, f, indent=4)