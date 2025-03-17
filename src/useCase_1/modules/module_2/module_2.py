#####################################
# Second Container 
# module_2.py --> automatic_speech_recognition.py
#  TCP Socket Client and also UDP client  
#####################################
# Libraries
import time
import logging
from otter_net_utils import OtterUtils
tcp_tools = OtterUtils()
print("OTTER Utils class imported successfully!")
# Specific librairies
import keras 
import tensorflow as tf
from transformers import pipeline
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from PIL import Image
#####################################

# CPU initialization
device = 'cpu'

#####################################
# Latent Space: Text/Image Query
class TextandImageQuery:
    QuestionText = None
    DescriptionText = None
    RawImage = None
    AnswerText = None

    def funcTextandImageQuery(self, raw_image_path):
        """
        Applies a neural network model to generate a textual description of an image.

        Args:
            raw_image_path (str): The path to the image file.

        Returns:
            None: The generated description is stored in `self.DescriptionText`.
        """
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        output = pipe(raw_image_path)
        self.DescriptionText = output[0]['generated_text']
    
    def funcQuestAnswering(self, question, text):
        """
        Performs question answering based on the provided text using a pre-trained DistilBERT model.

        Args:
            question (str): The question to be answered.
            text (str): The context or passage containing the answer.

        Returns:
            str: The predicted answer extracted from the text.
        """
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

        # Tokenize the question and text
        inputs = tokenizer(question, text, return_tensors="tf")
        outputs = model(**inputs)

        # Extract answer start and end positions
        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        # Decode the predicted answer
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        predict = tokenizer.decode(predict_answer_tokens)
        return predict

    def run(self):
        """
        Executes the image-to-text description and question-answering processes.

        This method first generates a textual description of the image stored in `self.RawImage` 
        using `funcTextandImageQuery`. Then, it performs question answering using 
        `funcQuestAnswering` based on `self.QuestionText` and the generated description.

        Returns:
            None: The answer is stored in `self.AnswerText`.
        """
        self.funcTextandImageQuery(self.RawImage)
        self.AnswerText = self.funcQuestAnswering(self.QuestionText, self.DescriptionText)

#####################################

if __name__ == "__main__":

    time.sleep(0.5)  # Wait during initialization of server containers
    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path) 

    ##################################### NETWORK PART #####################################
    # Network Parameters
    # C1 (Module 1 Container)
    HOST = 'module_1_container'  
    PORT = 5000             
    # C3 (Module 3 Container)
    HOST_3 = 'module_3_container' 
    PORT_3 = 5000

    ########### Interaction with C1 ###########
    s = tcp_tools.init_client_TCP_connection(HOST, PORT)
    # Receive the prediction from module_1
    buffer_size=1024
    data_decode = tcp_tools.wait_for_container_variable_TCP(s, buffer_size)

    ##########################################
    # Placeholder for future AI prediction 
    AIM_TIQ = TextandImageQuery()
    AIM_TIQ.QuestionText = data_decode
    AIM_TIQ.RawImage = "/app/inputs/image.png"
    AIM_TIQ.run()
    logging.info(f"Question Text: {AIM_TIQ.QuestionText}")
    logging.info(f"DESCRIPTION Text: {AIM_TIQ.DescriptionText}")
    logging.info(f"Answer Text: {AIM_TIQ.AnswerText}")
    prediction = AIM_TIQ.AnswerText
    prediction_encode = prediction.encode()
    ##########################################

    ########### Interaction with C3 ###########
    buffer_size = 1024
    s2, server_addr  = tcp_tools.init_client_connection_UDP(HOST_3, PORT_3, buffer_size)

    tcp_tools.send_variable_to_container_UDP(s2, prediction, server_addr) # Send Prediction 2 to C3