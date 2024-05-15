import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator

class QuizManager:
    ##########################################################
    def __init__(self, question_bank: list):
        """
        Task: Initialize the QuizManager class with a list of quiz questions.

        Overview:
        This task involves setting up the `QuizManager` class by initializing it with a list of quiz question objects. Each quiz question object is a dictionary that includes the question text, multiple choice options, the correct answer, and an explanation. The initialization process should prepare the class for managing these quiz questions, including tracking the total number of questions.

        Instructions:
        1. Store the provided list of quiz question objects in an instance variable named `questions`.
        2. Calculate and store the total number of questions in the list in an instance variable named `total_questions`.

        Parameters:
        - questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.

        Note: This initialization method is crucial for setting the foundation of the `QuizManager` class, enabling it to manage the quiz questions effectively. The class will rely on this setup to perform operations such as retrieving specific questions by index and navigating through the quiz.
        """
        ##### YOUR CODE HERE #####
        print("Task 9 initialized")
        self.questions = question_bank
        self.total_questions = len(self.questions)
        # pass # Placeholder
    ##########################################################

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.

        :param index: The index of the question to retrieve.
        :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
        """
        # Ensure index is always within bounds using modulo arithmetic
        valid_index = index % self.total_questions
        return self.questions[valid_index]
    
    ##########################################################
    def next_question_index(self, direction=None):
        """
        Task: Adjust the current quiz question index based on the specified direction.

        Overview:
        Develop a method to navigate to the next or previous quiz question by adjusting the `question_index` in Streamlit's session state. This method should account for wrapping, meaning if advancing past the last question or moving before the first question, it should continue from the opposite end.

        Instructions:
        1. Retrieve the current question index from Streamlit's session state.
        2. Adjust the index based on the provided `direction` (1 for next, -1 for previous), using modulo arithmetic to wrap around the total number of questions.
        3. Update the `question_index` in Streamlit's session state with the new, valid index.
            # st.session_state["question_index"] = new_index

        Parameters:
        - direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).

        Note: Ensure that `st.session_state["question_index"]` is initialized before calling this method. This navigation method enhances the user experience by providing fluid access to quiz questions.
        """
        new_index = st.session_state["question_index"] + direction
        st.session_state["question_index"] = new_index % self.total_questions
    


# Test Generating the Quiz
if __name__ == "__main__":
    
    path = os.path.abspath('../../')
    key_path = path + '\\radicalai-420811-b300c28fb7ea.json'
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "radicalai-420811",
        "location": "us-central1",
        "key_path": key_path
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config) 
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        questions = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write(f"Generating {questions} questions for the topic: '{topic_input}'  in Quiz Builder")
                # chroma_creator.create_chroma_collection()
                generator = QuizGenerator(topic_input, questions, chroma_creator.create_chroma_collection())
                question_bank = generator.generate_quiz()
                 

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            
            # Task 9
            ##########################################################
            quiz_manager = QuizManager(question_bank) # Use our new QuizManager class
            # Format the question and display
            with st.form("Multiple Choice Question"):
                index_question = quiz_manager.get_question_at_index(0) # Use the get_question_at_index method to set the 0th index
                
                # Unpack choices for radio
                choices = []
                for choice in index_question['choices']: # For loop unpack the data structure
                    
                    # Set the key from the index question
                    key = choice['key']
                    # Set the value from the index question
                    value = choice['value']
                    
                    choices.append(f"{key}) {value}")
                
                # Display the question onto streamlit
                st.write(index_question['question'])
                
                answer = st.radio( # Display the radio button with the choices
                    'Choose the correct answer',
                    choices
                )
                st.form_submit_button("Submit")
                
                if submitted: # On click submit 
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key): # Check if answer is correct
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
            ##########################################################