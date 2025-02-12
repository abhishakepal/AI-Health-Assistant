import streamlit as st
import nltk
# from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Another Model : 
# ----------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# ----------------------------------------------------------------------------------------------------------



# Load a pre-trained Hugging face model
# chatbot = pipeline("text-generation", model="distilgpt2")


# Define healthcare-specific response logic (or use a model to generate responses) 
# def healthcare_chatbot(user_input):
#     if "symptom" in user_input:
#         return "Please consult a Doctor for the accurate advice."
#     elif "appointment" in user_input:
#         return "Would you like to schedule an appointment with the doctor ?"
#     elif "medication" in user_input:
#         return "It's important to take prescribed medicines. If you have conserns, consult your doctor. "
#     else:
#         # For other inputs, use the Hugging Face model to generate a response 
#         response = chatbot(user_input,max_length = 300,num_return_sequences=1)
#         # Specifies the maximum length of the generated text response, including the input and the generated tokens. 
#         # If set to 3, the model generates three different possible responses based on the input. 
#     return response[0]['generated_text']




# Another Model's implementaion 
# ----------------------------------------------------------------------------------------------------------

def healthcare_chatbot(user_input):
    if "symptom" in user_input:
        return "Please consult a Doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines. If you have concerns, consult your doctor."
    else:
        # Generate response using Flan-T5
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# ----------------------------------------------------------------------------------------------------------





# Streamlit web app interface 
def main():
    # Set up the web app title and input area
    st.title("Healthcare Assistant Chatbot")

    # Display a simple text input for user queries
    user_input = st.text_input("How can I assist you today ?")

    # Display chatbot response 
    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, Please wait ...."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
            # print(response)
        else:
            st.write("Please enter a message to get a response.")
            
            
main()