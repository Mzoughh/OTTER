#####################################
# First Container 
# module_1.py --> automatic_speech_recognition.py
# TCP Socket SERVER Deployment 
#####################################
# Libraries
import logging
from otter_net_utils import OtterUtils
tcp_tools = OtterUtils()
print("OTTER Utils class imported successfully!")
# Specific librairies
import soundfile as sf
import torch
from transformers import pipeline
#####################################

# CPU initialization
device = 'cpu'

#####################################
# Speech Recognition Class
class AutomaticSpeechRecognition:
    QuestionAudio = None
    QuestionText = None

    def funcAutomaticSpeechRecognition(self, input):
        """
        Performs automatic speech recognition (ASR) on the given audio input.
        Args:
            input (str or audio file): The audio file or its path to be transcribed into text.
        Returns:
            str: The transcribed text from the audio.
        """
        speech_reco = pipeline(
            "automatic-speech-recognition", model="openai/whisper-base", device=device
        )
        res = speech_reco(input)
        return res["text"]

    def run(self):
        """
        Runs speech recognition on the stored audio input and saves the result.
        This method processes the audio stored in `self.QuestionAudio` using 
        the `funcAutomaticSpeechRecognition` method and stores the transcribed 
        text in `self.QuestionText`.
        """
        self.QuestionText = self.funcAutomaticSpeechRecognition(self.QuestionAudio)

#####################################

if __name__ == "__main__":

    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path)

    ##################################### NETWORK PART #####################################
    # Network Parameters
    HOST = '0.0.0.0'  # Listen on all network interfaces (Server configuration)
    PORT = 5000       # Port for listening
    conn = tcp_tools.init_server_TCP_connection(HOST, PORT)

    ##########################################
    AIM_ASR = AutomaticSpeechRecognition()
    AIM_ASR.QuestionAudio = "/app/inputs/OTTER_1_BIS.wav"  
    AIM_ASR.run()
    logging.info(f" AI prediction : Recognized Text = {AIM_ASR.QuestionText}")
    prediction = AIM_ASR.QuestionText
    ##########################################

    ##################################### NETWORK PART #####################################
    tcp_tools.send_variable_to_container_TCP(conn, prediction)

