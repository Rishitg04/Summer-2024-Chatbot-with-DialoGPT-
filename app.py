from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=['GET','POST'])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    # Create the attention mask
    attention_mask = torch.ones(new_user_input_ids.shape, dtype=torch.long)

    # Generate a response while limiting the total output to 1000 tokens
    response_ids = model.generate(
        new_user_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True,
        top_k=50,  
        top_p=0.95,  
        temperature=0.7  
    )

    # Decode the last output tokens from the bot
    response_text = tokenizer.decode(response_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response_text

    
if __name__ == '__main__':
    app.run(debug=False) 