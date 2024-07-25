# Generative AI Applications
## Introduction:
We are building some Generative AI Applications for creating some new contents.This system will provide a streamlit based user interface for user and gives the response to the user.

Mainly,Generative AI APPLICATIONS is a combination of  7 applications:
#### 1.	Search Engine
#### 2.	Document Q&A APPLICATION
#### 3.	Document Summarizer Application
#### 4.	Image Detector
#### 5.	Resume Application Tracking System
#### 6.	YouTube Transcript Generator
#### 7.	Health Management Application
This is an end to end LLM project using langchain   framework(which is specially useful  for developing applications powered by language model) based on some pretrained open source  LLM models which are:

•	llama3-8b-8192(developed by MetaAI),model type=Chat

•	neva-22b(developed by  NVIDIA),Model type=VLM(Visual Language Model)

•	Gemini-1.5-flash(developed by Google),model type=Chat,Vision and audio



## 1. Search Engine:
### Objective:
  The search engine is a python application that allows you to search any queries or question based on any topics.It will give the response to your queries.
### How It Works:
![image](https://github.com/user-attachments/assets/e527acf8-bebc-449b-99b3-c0d8d28dd623)
The application follows these steps to provide responses to your questions:
1.	Input Query Reading: The app reads your given query.
2.	Response Generation: Given query text and an input prompt are passed to the chat model(llama3-8b-8192),which generates a response based on the given query text and the input prompt.



## 2. Document Q&A Application:
### Objective:
  This is a python application that allows you to chat with multiple PDF documents.You can ask questions about the pdfs using natural language,and the application will provide relevant responses based on the 
  context of the documents.This apps utilizes a language model to generate accurate answers to your queries.Please note that the app will only respond to questions related to the loaded PDFs.
### How It Works:
![image](https://github.com/user-attachments/assets/ac9c0465-8ccc-4225-bf12-db2b737028a5)
The application follows these steps to provide responses to your questions:	
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Converting into Langchain Document:  These extracted texts are converted into langchain document.
3.	Document  Chunking:  The extracted langchain document  is divided into smaller chunks of texts(in the form of langchain document) that can be processed effectively.
4.	Embedding Model: The application utilizes a embedding model(embedding-001) to generate vector representations (embeddings) of those text chunks(which are in the form of langchain document) and store those chunks in a vector database FAISS that is also provided by langchain framework.
5.	Similarity Matching: When you ask a query, the app compares it with the text chunks(which are in the form of langchain document)  and identifies the most semantically similar ones.
6.	Response Generation: The selected chunks are passed to the chat model(llama3-8b-8192), which generates a response based on the input prompt and the relevant content of the PDFs.



## 3. Document Summarizer Application:
### Objective:
  This is a python application that allows you to summarize multiple PDF documents simultaneously.
### How It Works:
![image](https://github.com/user-attachments/assets/a29cf869-fb3e-4e7d-9437-0391bcdb7451)
The application follows these steps to provide summary of your uploaded pdfs:
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Text Chunking: The extracted text is divided into smaller chunks of text that can be processed effectively.
3.	Response Generation: The text chunks are passed to the chat model(llama3-8b-8192), which generates the summary responses one by one based on the number of text chunks and the input prompt.



## 4. Image Detector:
### Objective:
  This is a python application that allows you to detect the every articles in an image.
### How It Works:
![image](https://github.com/user-attachments/assets/5db63f74-fdb0-4041-b5de-f13b0e56e5dd)
The application follows these steps to detect the image:
1.	Image Loading: The app reads the uploaded image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for describing the uploaded image into a single message.
3.	Response Generation: Created message is passed into the visual language model(neva-22b) which generates the text about the every articles in the uploaded image.



## 5. Resume Application Tracking System:
### Objective:
  This is a python application that allows you to track your resume or cv based on any job description i.e. how much your resume or cv is eligible for a given job description.
### How It Works:
![image](https://github.com/user-attachments/assets/7dd85f42-8fc7-4a64-925a-5b317fead4d0)
The application follows these steps to track your resume or cv based on the given job description:
1.	Reading Job Description: The app reads the job description in the form of text.
2.	PDF Loading: The app reads your resume or cv PDF documents and extracts their text content.
3.	Response Generation: The extracted texts from the pdf document and the job description texts are passed to the chat model(llama3-8b-8192), which generates the response in a particular structured way based on the input prompt.



## 6. YouTube Transcript Generator:
### Objective:
This is a python application that allows you to generate the summarized transcription for a you tube video.
### How It Works:
![image](https://github.com/user-attachments/assets/e9350691-2d1e-4cb7-897f-71db20344041)
The application follows these steps to generate summarized transcription based on the you tube video:
1.	Reading YouTube video URL : The app reads the You Tube Video URL in the form of text.
2.	Extracting the Video ID and Generating the Transcript Text: Extract the video id from the text url and passed into the youtube_transcript _api function to generate the full transcript text of the you tube video.
3.	Response Generation: Extracted transcript text is passed into the chat model(llama3-8b-8192),which generates the summarized transcripted text in a structured way based on the input prompt.



## 7. Health Management Application:
### Objective:
  This is a python application that allows you to generate the total calories of a given food image and also tells you whether the food is healthy or not.
### How It Works:
![image](https://github.com/user-attachments/assets/767f2476-e499-43bd-aead-c827a3dac410)
The application follows these steps to generate the total calories of a given food image:
1.	Food Image Loading: The app reads the uploaded food image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for determining the calories of the uploaded food image into a single message.
3.	Response Generation: Created message is passed into the language model(Gemini-1.5-flash) which generates the response about the calories and the healthiness of the uploaded food image.



# Dependencies and Installation

1. Navigate to the project directory:

```bash
  cd genai app
```
2. Activate the conda environment:
```bash
  activate condaenv
```
3. Create a virtual environment in your local machine using:

```bash
  conda create -p venv python==3.10 -y
```
4. Activate the virtual environment:
```bash
  conda activate venv/
```
5. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
6. Acquire these three api keys through [GroqCloud](https://console.groq.com/playground),[Google AI Studio](https://aistudio.google.com/app/apikey) and [Nvidia](https://build.nvidia.com/explore/discover), then put them in .env file and keep this file in secret

```bash
  GROQ_API_KEY="your_api_key_here"
  GOOGLE_API_KEY="your_api_key_here"
  NVIDIA_API_KEY="your_api_key_here"
```

# Usage
1. Ensure that you have installed the required dependencies.
2. Run app.py by executing:
```bash
streamlit run app.py

```
3. The application will launch in your default browser,displaying the user interface.

## References

- [Langchain Documentation](https://python.langchain.com/v0.2/docs/introduction/)
