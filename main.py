#!/Users/enrique/opt/anaconda3/envs/gpt/bin/python

import cmd
import os
import glob
from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Initialize the OpenAI API
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
open_ai_key = os.getenv("OPENAI_API_KEY")


class Agent(object):

    def __init__(self):
        self.open_ai_key = None
        self.llm = None
        self.qa = None
        self.docs = []
        self.folder_path = "./doc"


    def bootstrap(self):
        self.initialize(open_ai_key)
        self.load_documentation()
        self.initialize_embeddings()

    def initialize(self, open_ai_key):
        self.open_ai_key = open_ai_key
        self.llm = ChatOpenAI(model = 'gpt-3.5-turbo', openai_api_key = open_ai_key, temperature=0)

    def load_documentation(self):
        # Walk through the folder and its subfolders
        for root, dirs, files in os.walk(self.folder_path):
            # Find all the html files in the current folder
            html_files = glob.glob(os.path.join(root, "*.html"))
            # Loop through the html files
            for html_file in html_files:
                try:
                    # Load up the file as a doc and split
                    loader = BSHTMLLoader(html_file)
                    self.docs.extend(loader.load_and_split())
                except Exception as e:
                    print(e)


    def initialize_embeddings(self):
        embeddings = OpenAIEmbeddings(openai_api_key = open_ai_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 400)
        pages = text_splitter.split_documents(self.docs)
        docsearch = FAISS.from_documents(pages, embeddings)

        # Create your retrieval engine
        self.qa = RetrievalQA.from_chain_type(llm = self.llm, chain_type = "stuff", retriever = docsearch.as_retriever())

    def query(self, prompt):
        query_result = self.qa.run(prompt)
        return query_result


class MyPrompt(cmd.Cmd):
    prompt = 'rabbithole > '
    intro = "Welcome! I am a FinOps agent. You can talk to me about anything FinOps. \nType ? to list commands"

    agent = None

    def default(self, line):
        query_result = self.agent.query(line)
        print('agent > %s' % query_result)

    def do_exit(self, inp):
        print("Bye")
        return True

if __name__ == '__main__':
    agent = Agent()
    agent.bootstrap()

    prompt = MyPrompt()
    prompt.agent = agent
    prompt.cmdloop()