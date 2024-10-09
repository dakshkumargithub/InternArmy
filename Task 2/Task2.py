from transformers import MarianMTModel, MarianTokenizer
import warnings

# Suppress FutureWarning for tokenization
warnings.filterwarnings("ignore", category=FutureWarning)

# Load English to Hindi Model
en_to_hi_model_name = 'Helsinki-NLP/opus-mt-en-hi'
en_to_hi_tokenizer = MarianTokenizer.from_pretrained(en_to_hi_model_name, clean_up_tokenization_spaces=True)
en_to_hi_model = MarianMTModel.from_pretrained(en_to_hi_model_name)

# Load Hindi to English Model
hi_to_en_model_name = 'Helsinki-NLP/opus-mt-hi-en'
hi_to_en_tokenizer = MarianTokenizer.from_pretrained(hi_to_en_model_name, clean_up_tokenization_spaces=True)
hi_to_en_model = MarianMTModel.from_pretrained(hi_to_en_model_name)

# Function to translate text using the specified model and tokenizer
def translate_text(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Generate translation
    translated_tokens = model.generate(**inputs)
    # Decode the translated tokens to get the final text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# English to Hindi translation
def english_to_hindi(text):
    return translate_text(text, en_to_hi_model, en_to_hi_tokenizer)

# Hindi to English translation
def hindi_to_english(text):
    return translate_text(text, hi_to_en_model, hi_to_en_tokenizer)

# CLI Interface for translation
def main():
    print("Choose Translation Direction:")
    print("1. English to Hindi")
    print("2. Hindi to English")
    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        text = input("Enter English text: ")
        translated = english_to_hindi(text)
        print("Translated to Hindi:", translated)
    elif choice == '2':
        text = input("Enter Hindi text: ")
        translated = hindi_to_english(text)
        print("Translated to English:", translated)
    else:
        print("Invalid choice. Please choose 1 or 2.")

if __name__ == '__main__':
    main()
