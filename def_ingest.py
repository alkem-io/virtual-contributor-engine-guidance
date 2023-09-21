import os
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import xml.etree.ElementTree as ET


from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import TextLoader

import shutil
import subprocess
import xml.etree.ElementTree as ET
from config import local_path, website_generated_path, vectordb_path, LOG_LEVEL

# configure logging
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(local_path+'/app.log')

c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
f_handler.setLevel(logging.ERROR)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

chunk_size=2000

def extract_urls_from_sitemap(base_directory):
    """
    Purpose:
        Read the sitemap.xml file and create a list of local html files to be read
    Args:
        base_directory: path to directory containing local html files
    Returns:
        list of files to be retrieved
    """

    sitemap_file = base_directory + os.sep + "sitemap.xml"
    logger.info(f"Extracting urls using {sitemap_file}")

    # Parse the XML directly from the file
    tree = ET.parse(sitemap_file)
    root = tree.getroot()

    # Extract the URLs from the sitemap
    to_be_retieved = [
        base_directory + elem.text + "index.html"
        for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    ]

    logger.debug(f"...sitemap as urls: {to_be_retieved[:5]}....")
    return to_be_retieved


def embed_text(texts, save_loc):
    embeddings = OpenAIEmbeddings(deployment=os.environ["AI_EMBEDDINGS_DEPLOYMENT_NAME"], chunk_size=1)
    docsearch = FAISS.from_documents(texts, embeddings)

    docsearch.save_local(save_loc)

def read_and_parse_html(local_source_path, source_website_url):
    """
    Purpose: read the target files from disk, transform html to readable text, remove sequnetial CR and space sequences, fix the document source address
             and split into chunks.
    Args:
        local_source_path: path to directory containing local html files
        source_website_url: base url of source website
    Returns: list of parses and split doucments
    """
    # Transform
    bs_transformer = BeautifulSoupTransformer()
    
    # Get all links from the sitemaps
    logger.info(f"generating html: {local_source_path}, {source_website_url}")
    full_sitemap_list = extract_urls_from_sitemap(website_generated_path)

    data = []
    for file_name in full_sitemap_list:
        loader = TextLoader(file_name)
        document = loader.load()
        doc_transformed = bs_transformer.transform_documents(document, tags_to_extract=["p", "article", "title", "h1"], remove_lines=True)
        body_text = doc_transformed[0]

        # first remove duplicate spaces, then remove duplicate '\n\n', then remove duplicate '\n \n '
        #body_text.page_content = re.sub(r'(\n ){2,}', '\n', re.sub(r'\n+', '\n', re.sub(r' +', ' ', body_text.page_content)))

        # remove the local directory from the source object
        body_text.metadata['source'] = body_text.metadata['source'].replace(local_source_path, source_website_url)

        data.append(body_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    logger.debug(texts)
    return texts

def remove_and_recreate(dir_path):
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Directory {dir_path} and its contents removed successfully.")
        os.makedirs(dir_path)
        logger.info(f"...directory {dir_path} (re)created.")
    except OSError as e:
        logger.error(f"Error: {e.strerror}")

def clone_and_generate(website_repo, destination_path, source_path):
    logger.info(f"About to generate website")
    remove_and_recreate(source_path)
    remove_and_recreate(destination_path)
    logger.info(f"...cloning or updating repo")

    # Check if the repository already exists in the source_path
    if os.path.exists(os.path.join(source_path, '.git')):
        # Repository exists, perform a git pull to update it
        git_pull_command = ['git', 'pull', 'origin', 'main']  # Modify branch name as needed
        result_pull = subprocess.run(git_pull_command, cwd=source_path, capture_output=True, text=True)
        logger.info(f"git pull result: {result_pull.stdout}")
    else:
        # Repository doesn't exist, perform a git clone
        clone_command = ['git', 'clone', website_repo, source_path]
        result_clone = subprocess.run(clone_command, capture_output=True, text=True)
        logger.info(f"git clone result: {result_clone.stdout}")

    os.chdir(source_path)
    logger.info(f"...cloned/updated, moved to directory: {os.getcwd()}")

    env = os.environ.copy()
    additional_path_go = '/usr/local/go/bin'
    additional_path_usr = '/usr/local'
    env["PATH"] = additional_path_go + os.pathsep + additional_path_usr + os.pathsep + env["PATH"]
    hugo_command = ['hugo', '--gc', '-b', '/', '-d', destination_path]
    result_hugo = subprocess.run(hugo_command, env=env, capture_output=True, text=True)
    logger.error(f"hugo result: {result_hugo.stdout}")


def mainapp(source_website_url) -> None:
    """
    Purpose:
        ingest the trnaformed website contents into a vector database in presized chunks.
    Args:
        source_website_url: full url of source website, used to return the proper link for the source documents.
    Returns:
        N/A
    """

    # open file to check output
    f = open(local_path+"/ingestion_output.txt", "w")

    # read and parse the files
    texts = read_and_parse_html(website_generated_path, source_website_url)

    # Save embeddings to vectordb
    embed_text(texts, vectordb_path)

    f.write(str(texts))
    f.close()


# only execute if this is the main program run (so not imported)
if __name__ == "__main__":
    mainapp(os.getenv('AI_SOURCE_WEBSITE'))
