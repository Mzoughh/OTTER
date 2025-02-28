#####################################
# Third Container
# module_3.py --> text_to_speech.py
# TCP Socket SERVER Deployment 
#####################################
#####################################
# Libraries
import logging
from otter_net_utils import OtterUtils
tcp_tools = OtterUtils()
print("OTTER Utils class imported successfully!")
# Specific librairies
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
#####################################

# CPU initialization
device = 'cpu'

#####################################
# Text to Speech conversion class
class TextToSpeech:
    """
    A class for converting text into speech using a neural TTS model.
    """
    AnswerText = None
    AnswerAudio = None

    def funcTextToSpeech(self, input_text):
        """
        Converts text into speech and saves the output as an audio file.

        Args:
            input_text (str): The text to be converted into speech.

        Returns:
            str: The path to the generated audio file.
        """
        # Initialize the text-to-speech pipeline with a pre-trained model
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)

        # Load speaker embeddings for voice synthesis
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Generate speech from text using the speaker embedding
        speech = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})

        # Define the output audio file path
        path_output = "/app/AudioAnswer.wav"

        # Save the generated speech as a WAV file
        sf.write(path_output, speech["audio"], samplerate=speech["sampling_rate"])

        return path_output

    def run(self):
        """
        Executes the text-to-speech conversion process.

        If `AnswerText` contains a valid text response, this method converts it to speech 
        and stores the resulting audio file path in `AnswerAudio`.

        Returns:
            None: The generated audio file path is stored in `self.AnswerAudio`.
        """
        if self.AnswerText:
            self.AnswerAudio = self.funcTextToSpeech(self.AnswerText)

#####################################

if __name__ == "__main__":
    
    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path)

    ##################################### NETWORK PART #####################################
    # Nerwork parameters
    HOST = '0.0.0.0'
    PORT = 5000  

    # Buffer size for send/receive variable
    buffer_size =  1024

    ########### Interaction with C2 ###########
    # Receive client request which ask for addr
    s, client_addr  = tcp_tools.init_server_UDP_connection(HOST, PORT, buffer_size)

    # Sent addr to C2
    tcp_tools.send_variable_to_container_UDP(s,'CON2 addr ?', client_addr)

    data_decode, _ = tcp_tools.wait_for_container_variable_UDP(s, buffer_size)

    ##########################################
    # Placeholder for future AI prediction 
    AIM_TTS = TextToSpeech()
    AIM_TTS.AnswerText = data_decode
    AIM_TTS.run()
    logging.info(f"Audio file created at: {AIM_TTS.AnswerAudio}")
    logging.info(f"End of deployment")
    ##########################################

