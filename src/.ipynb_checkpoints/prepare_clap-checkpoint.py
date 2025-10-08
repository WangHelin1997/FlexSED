import os
import pandas as pd
import torch
from transformers import AutoTokenizer, ClapTextModelWithProjection

if __name__ == '__main__':
    # Load the CLAP model and tokenizer
    model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    # Path to the input CSV file
    input_csv_path = '/home/user/SSD/Dataset/Audioset_SL/no_rule_all/label_to_id.csv'
    output_path = 'clap_embedding/'  # Replace with your desired output folder path

    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Get unique event labels
    events = df['label'].unique()

    with torch.no_grad():  # Disable gradient computation
        # Process each event
        for event in events:
            text = event.replace('_', ' ')  # Replace underscores with spaces
            text = f'The sound of {text}'
            print(text)
            inputs = tokenizer([text], padding=True, return_tensors="pt")
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds

            # Save the embeddings to a .pt file
            output_file = os.path.join(output_path, f"{event}.pt")
            torch.save(text_embeds, output_file)

        print("Embedding extraction and saving complete!")
