import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss


# Charger le jeu de données
dataset = load_dataset("joey234/mmlu-international_law-neg-prepend-fix")

# Configuration
model_id = "meta-llama/Llama-3.2-1B"

# Charger le tokenizer et le modèle
try:
    config_data = json.load(open('config.json'))
    my_secret_key = config_data['HF_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=my_secret_key)

    # Ajouter un token de padding si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Padding token ajouté au tokenizer.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=my_secret_key
    )
    model.resize_token_embeddings(len(tokenizer))  # Mettre à jour les poids pour inclure le padding
except Exception as e:
    raise ValueError("Erreur lors du chargement du modèle ou du tokenizer.") from e

# Vérifier que toutes les colonnes attendues sont présentes
required_columns = {"question", "choices", "answer"}
for split in ["test", "dev"]:
    if not required_columns.issubset(dataset[split].column_names):
        raise ValueError(f"Les colonnes requises {required_columns} sont manquantes dans le split {split}.")
print("Toutes les colonnes attendues sont présentes dans le dataset.")

# Prétraitement avec alignement dynamique
def preprocess_function(examples):
    input_texts = []
    labels = []

    for question, choice_list, answer in zip(examples["question"], examples["choices"], examples["answer"]):
        for i, choice in enumerate(choice_list):
            input_text = f"Question: {question} Choix: {choice}"
            tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)

            # Créer un label aligné avec input_ids
            label = tokenized["input_ids"] if i == answer else [-100] * len(tokenized["input_ids"])
            input_texts.append(tokenized["input_ids"])
            labels.append(label)

    # Convertir en tenseurs
    input_ids = torch.tensor(input_texts)
    labels = torch.tensor(labels)

    # Vérification d'alignement
    for idx, (label, input_id) in enumerate(zip(labels, input_ids)):
        if len(label) != len(input_id):
            print(f"Erreur d'alignement pour l'exemple {idx}:")
            print(f"Label: {label.tolist()}")
            print(f"Input IDs: {input_id.tolist()}")
        assert len(label) == len(input_id), \
            f"Les labels ({len(label)}) ne sont pas alignés avec les input_ids ({len(input_id)})."

    return {"input_ids": input_ids, "labels": labels}

# Test sur un sous-ensemble
print("Test sur un sous-ensemble des données...")
subset = dataset["test"].select(range(10))  # Prendre seulement les 10 premiers exemples pour valider
tokenized_subset = subset.map(preprocess_function, batched=True, remove_columns=subset.column_names)
print("Validation sur sous-ensemble réussie !")

# Étendre le test à l'ensemble des données
print("Traitement de l'ensemble des données...")
tokenized_train = dataset["test"].map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names)
tokenized_eval = dataset["dev"].map(preprocess_function, batched=True, remove_columns=dataset["dev"].column_names)
print("Traitement complet des données terminé.")

# Calcul manuel de la perte pour un échantillon
print("Calcul manuel de la perte pour un échantillon...")
sample_input_ids = torch.tensor(tokenized_train["input_ids"][0]).unsqueeze(0)  # Convertir en tenseur et ajouter une dimension batch
sample_labels = torch.tensor(tokenized_train["labels"][0]).unsqueeze(0)        # Idem pour les labels
outputs = model(input_ids=sample_input_ids, labels=sample_labels)
loss_fn = CrossEntropyLoss(ignore_index=-100)
logits = outputs.logits
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = sample_labels[:, 1:].contiguous()
loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(f"Perte calculée pour un échantillon : {loss.item()}")

# Recherche du meilleur learning rate
def find_learning_rate():
    lr_range = torch.logspace(-7, -1, steps=50).tolist()  # Plage logarithmique
    losses = []

    for lr in lr_range:
        training_args = TrainingArguments(
            output_dir="./results_lr_finder",
            learning_rate=lr,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="no",  # Pas de sauvegarde pendant la recherche
            logging_dir=None,
            logging_steps=1,     # Enregistrer chaque étape
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train.select(range(10)),  # Sous-ensemble pour la recherche rapide
        )

        trainer.train()

        # Récupérer la dernière perte directement depuis l'état
        log_history = trainer.state.log_history
        loss = None
        for log in reversed(log_history):  # Parcourir les logs pour trouver une perte valide
            if "loss" in log:
                loss = log["loss"]
                break
        
        if loss is None:
            print(f"Erreur : impossible de récupérer la perte pour lr={lr}")
            losses.append(float('nan'))
        else:
            print(f"Learning rate: {lr}, Loss: {loss}")
            losses.append(loss)

    # Tracer la courbe
    plt.figure(figsize=(8, 6))
    plt.plot(lr_range, losses, marker='o')
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid()
    plt.show()


# Étape 4 : Configurer l'entraînement complet avec gradient clipping et métriques
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.05689866095781326,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",  # Sauvegarde périodique
    max_grad_norm=1.0,  # Gradient clipping
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()

# Sauvegarde du modèle
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")