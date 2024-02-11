import base64
import requests
import cv2
from gtts import gTTS
from pydub import AudioSegment
import os
from dotenv import load_dotenv,find_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import time

from PIL import Image
import io
import gradio as gr
import subprocess
load_dotenv(find_dotenv())
stabilityai_key=os.getenv("stabilityai_api")
openai_key=os.getenv("openai_api")
os.environ["OPENAI_API_KEY"] = openai_key
huggingface_key=os.getenv("huggingface_key")

def merge_audio_files( output_file='D:\\python\\Aiapp\\audio.flac',input_folder="D:\\python\\Aiapp\\audio_file"):
     files = os.listdir(input_folder)
     num_files = len(files)
     files = [os.path.join(input_folder, file) for file in files]
     merged_audio = AudioSegment.from_file(files[0])
     for i in range(1,num_files):
         merged_audio+=AudioSegment.from_file(files[i])
     merged_audio.export(output_file, format="mp3")
def delete_audiofiles(folder_path='D:\\python\\Aiapp\\audio_file'):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def get_audio_duration(index):
   audio_file=rf'D:\python\Aiapp\audio_file\audio_{index}.flac'
   audio = AudioSegment.from_file(audio_file)
   duration_ms = len(audio)
   duration_seconds = duration_ms / 1000.0
   return duration_seconds

def merge():
    command = 'C:\\Users\\Selva\\ffmpeg\\ffmpeg-6.1.1-essentials_build\\bin\\ffmpeg.exe -y -i D:\\python\\Aiapp\\output_video.mp4 -i D:\\python\\Aiapp\\audio.flac D:\\python\\Aiapp\\output.mp4'
    subprocess.run(command, shell=True)






    
    
def promptgenerator(scene):
    template="""
    you are given a scene from a story ;genearate a prompt for image generation,image describes the scene;
    ;SCENE:{scene}
    prompt:"""
    prompt=PromptTemplate(template=template,input_variables=["scene"])
    prompt_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    prompt=prompt_llm.predict(scene=scene)
    print(prompt)
    return prompt
def generatestory(context):
    template="""
    you are a story teller;genearate a creative story about the prompt,max  40 words;
    CONTEXT:{context}
    STORY:"""
    
    prompt=PromptTemplate(template=template,input_variables=["context"])
    story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    story=story_llm.predict(context=context)
    print(story)
    return story


def images_to_video(imagelist, video_name='D:\python\Aiapp\output_video.mp4', fps=24):
    duplicated_images =imagelist
    frame = cv2.imread(duplicated_images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    for image in duplicated_images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
def text_to_audio(text,index):
    if text:
        tts=gTTS(text)
        tts.save(rf'D:\python\Aiapp\audio_file\audio_{index}.flac')
    else:
        tts=gTTS("The end")
        tts.save(rf'D:\python\Aiapp\audio_file\audio_{index}.flac')

def text2img(prompt,index):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    body = {
  "steps": 40,
  "width": 1024,
  "height": 1024,
  "seed": 0,
  "cfg_scale": 5,
  "samples": 1,
  "text_prompts": [
    {
      "text": f"{prompt}",
      "weight": 1
    },
    {
      "text": "blurry, bad",
      "weight": -1
    }
  ],
}
    headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {stabilityai_key}",
}
    response = requests.post(
  url,
  headers=headers,
  json=body,
)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    if not os.path.exists("D:\python\Aiapp\out"):
        os.makedirs("D:\python\Aiapp\out")
    for image in data["artifacts"]:
        with open(f'D:\python\Aiapp\out/txt2img_{index}.png', "wb") as f:
            f.write(base64.b64decode(image["base64"]))

def func(story):
    
    result_list=story.split(".")
    result_list=[s.strip() for s in result_list]
    
    print(result_list)
    imagelist=[]
    fps=24
    delete_audiofiles()
    for i in range(len(result_list)):
        time.sleep(5)
        text_to_audio(text=result_list[i],index=i)
        prompt=promptgenerator(result_list[i])
        text2img(prompt=prompt,index=i)
        duration=get_audio_duration(index=i)
        imagelist.extend([f'D:\\python\\Aiapp\out\\txt2img_{i}.png']*round(fps*duration))
    images_to_video(imagelist)
    merge_audio_files()
    merge()

def video(input_prompt):
    story=generatestory(input_prompt)
    time.sleep(20)
    func(story)
    path='D:\\python\\Aiapp\\output.mp4'
    return gr.Video(path)
def video_by_story(story):
    func(story)
    path='D:\\python\\Aiapp\\output.mp4'
    return gr.Video(path)
    
def main():
    with gr.Blocks() as demo:
        txt1 = gr.Textbox(label="Input prompt")
        
        btn1 = gr.Button(value="Submit")
        output=gr.Video()
        btn1.click(video, inputs=[txt1], outputs=[output])
        txt2=gr.Textbox(label="input your own story",lines=5)
        btn2=gr.Button(value="submit")
        btn2.click(video_by_story, inputs=[txt2], outputs=[output])
        
    demo.launch()
    
    
if __name__=='__main__':
    main()










   




   
