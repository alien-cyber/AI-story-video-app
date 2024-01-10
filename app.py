import base64
import requests
import cv2
import os
from dotenv import load_dotenv,find_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import time
import streamlit as st
from PIL import Image
import io
import gradio as gr
import subprocess
def merge():
    command = 'C:\\Users\\Selva\\ffmpeg\\ffmpeg-6.1.1-essentials_build\\bin\\ffmpeg.exe -y -i D:\\python\\Aiapp\\output_video.mp4 -i D:\\python\\Aiapp\\audio.flac D:\\python\\Aiapp\\output.mp4'
    subprocess.run(command, shell=True)
load_dotenv(find_dotenv())
stabilityai_key=os.getenv("stabilityai_api")
openai_key=os.getenv("openai_api")
os.environ["OPENAI_API_KEY"] = openai_key
huggingface_key=os.getenv("huggingface_key")


def deleteimages():
    folder_path = 'D:\python\out'
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def spiltstory(story):
    template="""
     you are given a story;split the story into some parts,example you are given
     a story a man goes to the forest and he finds a reasure
     Scene:a man goes to the forest Scene:he finds a treasure
     CONTEXT:{story}
     SPLITED SENTENCE:
     """
    prompt=PromptTemplate(template=template,input_variables=["story"])
    prompt_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    prompt=prompt_llm.predict(story=story)
    
    return prompt
def promptgenerator(scene):
    template="""
    you are given a scene from a story ;genearate a prompt for image generation,image describes the scene;
    the promth should not have details like name,place one exampe prompth 
    EXAMPLEPROMPT:(impressionistic realism by csle, working in banking, very shybgh), a 50 something maort dyed dark curly balding hair, Afro-Asiatic ancestry, talks a lot but listens poorly, stuck in the past, wearing a suit, he has a certain charm, bronze skintone, sitting in a bar at night, he is smoking and feeling cool, drunk on plum wine, masterpiece, 8k, hyper detailed, smokey ambiance, perfect hands AND fingers;
    CONTEXT:{scene}
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

def images_to_video(image_folder, video_name='D:\python\Aiapp\output_video.mp4', fps=24, duration_per_image=3):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Duplicate each image according to the desired duration
    duplicated_images = []
    for image in images:
        duplicated_images.extend([image] * int(fps * duration_per_image))

    frame = cv2.imread(os.path.join(image_folder, duplicated_images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))


    for image in duplicated_images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
def text_to_audio(text):
    
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {huggingface_key}",}
    payload={"inputs":text}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open(r'D:\python\Aiapp\audio.flac','wb') as file:
        file.write(response.content)




def text2img(prompt,index):
    API_URL = "https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1"
    headers = {"Authorization": f"Bearer {huggingface_key}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
    "inputs": f"{prompt}",
})
    if not os.path.exists("D:\python\Aiapp\out"):
        os.makedirs("D:\python\Aiapp\out")
    image = Image.open(io.BytesIO(image_bytes))
    output_filename = f'D:\\python\\Aiapp\out\\txt2img_{index}.png'
    image.save(output_filename)


def video(input_prompt):
    
    
    story=generatestory(input_prompt)
    time.sleep(20)
    def func(story):
        spiltsentence=spiltstory(story)
        result_list=spiltsentence.split("Scene")
        result_list=[s.strip() for s in result_list]
        result_list=result_list[1:]
        print(result_list)
        for i in range(len(result_list)):
            time.sleep(5)
            prompt=promptgenerator(result_list[i])
            text2img(prompt=prompt,index=i)
        images_to_video('D:\python\Aiapp\out')
        text_to_audio(story)
        merge()
            

            
        path='D:\\python\\Aiapp\\output.mp4'
        return gr.Video(path)
    func(story)
    
        

def main():
    with gr.Blocks() as demo:
        txt = gr.Textbox(label="Input prompt")
        btn = gr.Button(value="Submit")
        output=gr.Video()
        btn.click(video, inputs=[txt], outputs=[output])
    demo.launch()
    
    
if __name__=='__main__':
    main()










   
