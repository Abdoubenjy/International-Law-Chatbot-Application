# **Business Chatbot - Fine-Tuning and User Interface** 🤖💬

This project involves fine-tuning a causal language model (**Meta Llama 3.2 - 1B**) on a custom dataset and deploying it through an interactive **Streamlit** user interface. The app allows users to ask questions and receive generative responses from the fine-tuned model.

---

## **Project Contents** 📂

### **1. Main Scripts**
1. **Fine-Tuning (`fine_tuning.py`)**:
   - Fine-tunes the model on a specified dataset.
   - Key steps:
     - Dataset loading and preprocessing.
     - Dynamic label alignment and tokenization.
     - Learning rate optimization.
     - Full model training with metric evaluation.
     - Model and tokenizer saving.

2. **User Interface (`app.py`)**:
   - A **Streamlit-based** web app for interacting with the fine-tuned model.
   - Features:
     - Input field for questions.
     - Clean, direct generative responses displayed to the user.

---

## **Installation** 🛠️

### **Prerequisites**
1. **Python 3.7 or later**
2. **Required Packages**:
   - Install dependencies with:
     ```bash
     pip install -r requirements.txt
     ```
   - Example `requirements.txt`:
     ```plaintext
     torch
     transformers
     datasets
     streamlit
     matplotlib
     ```

### **Project Structure** 📁
```plaintext
.
├── app.py                 # Streamlit app script
├── fine_tuning.py         # Model fine-tuning script
├── config.json            # File with HF_TOKEN for Hugging Face access
├── requirements.txt       # Required dependencies
├── test_fine_tuned_model/ # Directory for the fine-tuned model
```

# Usage 🚀 

## 1. Fine-Tuning the Model

### Configure the `config.json` file:

Create a `config.json` file with the following content:

```json
{
  "HF_TOKEN": "your_hugging_face_token"
}
```

Replace `"your_hugging_face_token"` with your personal Hugging Face token.

### Run the fine-tuning script:

```bash
python fine_tuning.py
```

The script will:
- Download and process the dataset (`joey234/mmlu-international_law-neg-prepend-fix`).
- Fine-tune the model.
- Save the fine-tuned model and tokenizer to the `./fine_tuned_model` directory.

### Validation:

Ensure the required files (`pytorch_model.bin`, `config.json`, etc.) are present in `./fine_tuned_model`.

## 2. Launch the User Interface

### Ensure the fine-tuned model is ready:

Move the fine-tuned model files into a folder named `test_fine_tuned_model`.

### Run the Streamlit app:

```bash
streamlit run app.py
```

This will open an interactive interface in your browser.

- Input your question into the provided field and receive a generative response.

---

## Key Features ✨

### Fine-Tuning

- Loads and preprocesses a custom dataset.
- Dynamic label alignment with tokenization.
- Learning rate optimization with visualization.
- Full training with metric evaluation and gradient clipping.

### User Interface

- Simple question-and-answer functionality.
- Clean, context-aware generative responses.
- Easy-to-use interface, accessible via a web browser.

