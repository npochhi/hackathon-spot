import os
import time
from spot_controller import SpotController
import cv2
import requests
from openai import OpenAI
import mimetypes
import base64
import json
import sys


ROBOT_IP = "192.168.50.3"#os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"#os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"#os.environ['SPOT_PASSWORD']

api_key = "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"
client = OpenAI(api_key=api_key)


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string with MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def construct_payload(images: list[str], prompt: str, role="user", model="gpt-4-vision-preview", max_tokens=1000):
    """Construct the payload for the GPT-4 Vision API request."""
    encoded_images = [encode_image_to_base64(image) for image in images]
    content = [{"type": "text", "text": prompt}] + \
              [{"type": "image_url", "image_url": {"url": image}} for image in encoded_images]
    messages = [{"role": role, "content": content}]
    return {"model": model, "messages": messages, "max_tokens": max_tokens}

def query_openai(payload: dict):
    """Send a request to the OpenAI API and return the response."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def text_to_speech(text="Hello, world!", model="tts-1", voice="alloy", output_file="output.mp3"):
    """Convert text to speech and save as an MP3 file."""
    response = client.audio.speech.create(model=model, voice=voice, input=text)
    response.stream_to_file(output_file)

def speech_to_text(audio_file="./audio.mp3", model="whisper-1"):
    """Convert speech from an audio file to text."""
    transcript = client.audio.transcriptions.create(model=model, file=audio_file)
    return transcript

def main():
    camera_capture = cv2.VideoCapture(0)

    # Use wrapper in context manager to lease control, turn on E-Stop, power on robot and stand up at start
    # and to return lease + sit down at the end
    with SpotController(username=SPOT_USERNAME, password=SPOT_PASSWORD, robot_ip=ROBOT_IP) as spot:
        print("Playing sound")
        os.system(f"ffplay -nodisp -autoexit -loglevel quiet {"intro_spot.mp3"}")

        time.sleep(2)
        sample_name = "aaaa.mp3"
        cmd = f'arecord -vv --format=cd --device={os.environ["AUDIO_INPUT_DEVICE"]} -r 48000 --duration=10 -c 1 {sample_name}'
        os.system(cmd)
        
        # prompt_text = speech_to_text(audio_file="aaaa.mp3")
        prompt_text = "Tell me something cool about this room?"


        _, center_image = camera_capture.read()
        camera_capture.release()
        cv2.imwrite('center_image.png', center_image)

        # Move head to specified positions with intermediate time.sleep
        spot.move_head_in_points(yaws=[0.8, 0],
                                 pitches=[0, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        _, left_image = camera_capture.read()
        camera_capture.release()
        cv2.imwrite('left_image.png', left_image)

        time.sleep(3)

        spot.move_head_in_points(yaws=[-0.8, 0],
                                 pitches=[0, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        _, right_image = camera_capture.read()
        camera_capture.release()

        cv2.imwrite('right_image.png', right_image)
        
        time.sleep(3)

        spot.move_head_in_points(yaws=[0, 0],
                                 pitches=[0.8, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        _, top_image = camera_capture.read()
        camera_capture.release()

        cv2.imwrite('top_image.png', top_image)

        time.sleep(3)

        spot.move_head_in_points(yaws=[0, 0],
                                 pitches=[-0.8, 0],
                                 rolls=[0, 0],
                                 sleep_after_point_reached=1)
        
        _, bottom_image = camera_capture.read()
        camera_capture.release()
        
        cv2.imwrite('bottom_image.png', bottom_image)

        time.sleep(3)

        image_paths = ["./left_image.png", "./right_image.png"]
        sys_prompt = "You are a robot dog, and a real estate agent in San Francisco. Write a simple, VERY BRIEF, humorous pitch for this room."
        
        user_prompt = prompt_text
        combined_prompt = f"{sys_prompt} The user just said: {user_prompt}"
        payload = construct_payload(images=image_paths, prompt=combined_prompt)
        response = query_openai(payload)

        answer = response["choices"][0]["message"]["content"]

        text_to_speech(answer)

        os.system(f"ffplay -nodisp -autoexit -loglevel quiet output.mp3")

        time.sleep(2)


if __name__ == '__main__':
    main()
