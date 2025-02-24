import streamlit as st
from transformers import pipeline, AutoTokenizer, QuestionAnsweringPipeline
from src.models.modernbert.modeling_modernbert import ModernBertForQuestionAnswering

MODEL_PATH = 'kiddothe2b/ModernBERT-base-squad2'


class CustomQuestionAnsweringPipeline(QuestionAnsweringPipeline):
    def check_model_type(self, supported_models):
        """Hack: Overriding function which checks if the model type is natively supported by HuggingFace"""
        if self.model.__class__.__name__ != 'ModernBertForQuestionAnswering':
            print("Only ModernBERT is supported!")


# Load HuggingFace pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, token=True, padding_side='right')
model = ModernBertForQuestionAnswering.from_pretrained(MODEL_PATH, token=True)

# Load the question-answering pipeline with a custom pipeline to avoid errors
qa_engine = pipeline(task="question-answering", model=model, tokenizer=tokenizer,
                     pipeline_class=CustomQuestionAnsweringPipeline)

# Streamlit app header
st.markdown("# **Question Answering with ModernBERT**")

# Input for context
context = st.text_area("**Context**", "Enter a short paragraph as a source for your question...")

# Input for question
question = st.text_input("**Question**", "Enter your question here...")

# Button to get the answer
if st.button("Get Answer"):
    if context and question:
        # Use the pipeline to get the answer
        answer = qa_engine(question=question, context='Document: ' + context)
        # If score probability low, opt to not answer
        st.write("**Answer**:", answer['answer'])
    else:
        st.warning("Please fill in both input fields (context paragraph and question).")
