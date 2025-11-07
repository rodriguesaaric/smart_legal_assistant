import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json
import random
from pathlib import Path

def train_spacy_model(
    data_path=Path(__file__).parent / "train.json",
    model_output_dir=Path(__file__).parent / "clause-model",
    base_model="en_core_web_sm"
):
    """
    Trains a spaCy NER model by updating a pre-trained model.
    This is more effective than training from blank.
    """
    print("Starting model training...")

    # 1. Load the base model
    try:
        nlp = spacy.load(base_model)
        print(f"Loaded base model '{base_model}'")
    except IOError:
        print(f"Error: Base model '{base_model}' not found.")
        print(f"Run: python -m spacy download {base_model}")
        return

    # 2. Load your training data
    with open(data_path, 'r', encoding='utf-8') as f:
        TRAIN_DATA = json.load(f)

    # 3. Add the NER (Named Entity Recognizer) to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Add our new label "CLAUSE_START"
    ner.add_label("CLAUSE_START")

    # 4. Convert data to spaCy's format
    db = DocBin()
    examples = []
    print("Processing training data...")
    for text, annots in TRAIN_DATA:
        try:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annots["entities"]:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    print(f"Warning: Skipping entity for '{text[start:end]}' in '{text}'")
                else:
                    ents.append(span)
            
            example = Example.from_dict(doc, {"entities": ents})
            examples.append(example)
        except Exception as e:
            print(f"Error processing: {text} | {e}")
    
    print(f"Created {len(examples)} training examples.")

    # 5. Start the training
    print("Initializing training...")
    
    # Disable other pipes during training for efficiency
    disabled_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    with nlp.select_pipes(disable=disabled_pipes):
        optimizer = nlp.initialize(lambda: examples)

        # 6. Train the model
        n_iter = 30  # Increased iterations for better learning
        print(f"Training for {n_iter} iterations...")
        
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            
            # This is the main training loop
            nlp.update(examples, drop=0.3, losses=losses, sgd=optimizer)
            
            if itn % 10 == 0:
                print(f"Iteration {itn}/{n_iter}, Losses: {losses}")

    # 7. Save the trained model to disk
    nlp.to_disk(model_output_dir)
    print(f"\nSuccess! Model trained and saved to '{model_output_dir}'")
    print("You can now use this model in your main application.")

if __name__ == "__main__":
    train_spacy_model()