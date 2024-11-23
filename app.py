from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load the paraphrasing model and tokenizer
model_name = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase_text(text, max_length=100, num_return_sequences=1):
    """
    Generate a paraphrased version of the given text with optimized processing time.
    
    Args:
    - text (str): Input text to paraphrase.
    - max_length (int): Maximum length of the paraphrased text.
    - num_return_sequences (int): Number of paraphrases to generate (set to 1 for speed).
    
    Returns:
    - list: A list of paraphrased versions of the input text, one per paragraph.
    """
    paragraphs = text.split("\n")  # Split the text by paragraphs (assuming paragraphs are separated by newlines)
    paraphrased_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph.strip():  # Avoid paraphrasing empty paragraphs
            # Prepend the task prefix (specific to T5 models)
            input_text = f"paraphrase: {paragraph} </s>"
            
            # Tokenize the input text
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)
            
            # Generate paraphrased outputs
            outputs = model.generate(
                inputs,
                max_length=max_length,           # Reduced maximum length
                num_beams=3,                     # Reduced beam size to 3 for faster results
                temperature=1.0,                  # Controls creativity; still maintaining a reasonable variety
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=2           # Helps avoid repetitive text while keeping output diverse
            )
            
            # Decode the output into readable text
            paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrased_paragraphs.append(paraphrased_text)
    
    return paraphrased_paragraphs

@app.route('/', methods=['GET', 'POST'])
def home():
    paraphrased_paragraphs = []
    if request.method == 'POST':
        original_text = request.form['article']
        paraphrased_paragraphs = paraphrase_text(original_text)
    return render_template('index.html', paraphrased_paragraphs=paraphrased_paragraphs)

if __name__ == "__main__":
    app.run(debug=True)
