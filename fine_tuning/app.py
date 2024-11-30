import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Charger le modèle et le tokenizer
model_dir = "./test_fine_tuned_model"

@st.cache_resource
def load_model():
    """
    Charge le modèle et le tokenizer.
    Utilise Streamlit cache pour éviter de recharger plusieurs fois.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to("cpu")  # Assurez-vous que tout est sur le CPU
    return model, tokenizer

# Charger le modèle et le tokenizer
model, tokenizer = load_model()

# Fonction pour générer une réponse
def generate_answer(question):
    """
    Génère une réponse à partir d'une question en utilisant le modèle génératif.
    Nettoie la réponse générée pour exclure la question ou les répétitions.
    """
    # Préparer l'entrée pour le modèle
    input_text = f"Question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Générer une réponse
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

    # Décoder la réponse générée
    raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Nettoyer la réponse pour retirer la question et autres parties inutiles
    cleaned_answer = raw_answer.replace("Question:", "").strip()
    # Supprimer la question elle-même si elle apparaît dans la réponse
    if question in cleaned_answer:
        cleaned_answer = cleaned_answer.replace(question, "").strip()
    # Supprimer les répétitions ou contenu superflu
    cleaned_answer = cleaned_answer.replace("Choix:", "").strip()

    return cleaned_answer

# Interface utilisateur avec Streamlit
st.title("Business Chatbot - Réponse Générative")
st.write("Posez une question, et le modèle générera une réponse.")

# Champ d'entrée pour la question
question = st.text_input("Posez votre question", placeholder="Entrez votre question ici...")

# Bouton pour générer une réponse
if st.button("Obtenir une réponse"):
    if question:
        # Générer la réponse
        answer = generate_answer(question)
        if answer:
            st.success(answer)  # Afficher uniquement la réponse nettoyée
        else:
            st.warning("La réponse générée semble vide après nettoyage.")
    else:
        st.error("Veuillez entrer une question.")
