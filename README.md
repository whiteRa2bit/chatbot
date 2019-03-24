# chatbot
Educational project

It's a russian chatbot trained using seq2seq model with attention mechanism. 
To use the chatbot install the packages listed in requirments.txt, then launch the file model.py

Then initialize in your project objects:
```
with open('./data/seq2seq.pk', 'rb') as f:
        data = pickle.load(f)
    input_lang = data['input_lang']
    output_lang = data['output_lang']
    encoder = data['encoder']
    attn_decoder = data['attn_decoder']
    hidden_size = encoder.hidden_size
```
then just call function 
```
get_answer(input_sentence, input_lang, output_lang, encoder, attn_decoder)
```
