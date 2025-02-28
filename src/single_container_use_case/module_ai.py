#####################################
# First Container 
# module_ai.py
# Deployment of a container
#####################################
# Libraries
import logging
import numpy # As an example of requirement.txt import
# Specific librairies
import soundfile as sf
import torch
from transformers import pipeline
#####################################

# CPU initialization
device = 'cpu'

def build_log_file(log_file_path):
    """
    Creates an empty log file at the specified path and configures logging settings.
    Args:
        log_file_path (str): The path where the log file will be created.

    Returns:
        None
    """
    with open(log_file_path, 'w') as file:
        pass  # Create an empty file

    logging.basicConfig(
        filename=log_file_path, 
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

#####################################
# Speech Recognition Class
class AutomaticSpeechRecognition:
    QuestionAudio = None
    QuestionText = None

    def funcAutomaticSpeechRecognition(self, input):
        """
        Performs automatic speech recognition (ASR) on an audio file.
        Args: : 
            The audio file 
        Returns:
        str:
            The transcribed text from the audio.
        """
        speech_reco = pipeline(
            "automatic-speech-recognition", model="openai/whisper-base", device=device
        )
        res = speech_reco(input)
        return res["text"]

    def run(self):
        """
        Runs speech recognition on the audio stored in self.QuestionAudio
        and stores the transcription in self.QuestionText.
        """
        self.QuestionText = self.funcAutomaticSpeechRecognition(self.QuestionAudio)
#####################################

if __name__ == "__main__":

    # Log Initialization
    log_file_path = "/app/logs/container.log"
    build_log_file(log_file_path)
    
    ##################################### NETWORK PART #####################################
    
    #####################################
    # Placeholder for future Network parameter
    #####################################

    ##########################################
    AIM_ASR = AutomaticSpeechRecognition()
    AIM_ASR.QuestionAudio = "/app/inputs/OTTER_1_BIS.wav"  
    AIM_ASR.run()
    logging.info(f" AI prediction : Recognized Text = {AIM_ASR.QuestionText}")
    prediction = AIM_ASR.QuestionText
    ##########################################