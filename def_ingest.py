import os
import logging
import sys
import io
import re
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
import xml.etree.ElementTree as ET


from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import TextLoader

import shutil
import subprocess
import xml.etree.ElementTree as ET
from config import config, local_path, website_generated_path, website_generated_path2, vectordb_path, website_source_path, website_source_path2, github_user, github_pat, github_pat, LOG_LEVEL, chunk_size

# configure logging
logger = logging.getLogger(__name__)
assert LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
logger.setLevel(getattr(logging, LOG_LEVEL))  # Set logger level


# Create handlers
c_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, line_buffering=True))
f_handler = logging.FileHandler(os.path.join(os.path.expanduser(local_path), 'app.log'))

c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
f_handler.setLevel(logging.WARNING)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info(f"log level {os.path.basename(__file__)}: {LOG_LEVEL}")

def create_sitemap_filepath(website_generated_path):
    return website_generated_path + os.sep + "sitemap.xml"

def sitemap_file_exists(website_generated_path):
    sitemap_file = create_sitemap_filepath(website_generated_path)
    return os.path.exists(sitemap_file)

def extract_urls_from_sitemap(base_directory):
    """
    Purpose:
        Read the sitemap.xml file and create a list of local html files to be read
    Args:
        base_directory: path to directory containing local html files
    Returns:
        list of files to be retrieved
    """

    sitemap_file = create_sitemap_filepath(base_directory)
    logger.info(f"Extracting urls using {sitemap_file}")

    # Parse the XML directly from the file
    tree = ET.parse(sitemap_file)
    root = tree.getroot()

    # List to store the complete URLs of the webpages to be retrieved
    webpages_to_retrieve = []
    # Extract the URLs from the sitemap
    for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        # replace the / with the os separator
        url_path = elem.text.replace("/", os.sep)
        complete_url = base_directory + url_path + "index.html"
        webpages_to_retrieve.append(complete_url)

    # logger.info(f"...sitemap as urls: {webpages_to_retrieve[:5]}....")
    return webpages_to_retrieve


def embed_text(texts, save_loc):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config['embeddings_deployment_name'],
        openai_api_version=config['openai_api_version'],
        chunk_size=1
    )
    docsearch = FAISS.from_documents(texts, embeddings)

    docsearch.save_local(save_loc)

def read_and_parse_html(local_source_path, source_website_url, website_generated_path):
    """
    Purpose: read the target files from disk, transform html to readable text, remove sequnetial CR and space sequences, fix the document source address
             and split into chunks.
    Args:
        local_source_path: path to directory containing local html files
        source_website_url: base url of source website
        website_generated_path: path to directory containing generated html files
    Returns: list of parses and split doucments
    """
    # Transform
    bs_transformer = BeautifulSoupTransformer()

    # Get all links from the sitemaps
    logger.info(f"generating html: {local_source_path}, {source_website_url}")
    full_sitemap_list = extract_urls_from_sitemap(website_generated_path)

    exclusion_list = [os.sep + 'tag' + os.sep,
                      os.sep + 'category' + os.sep,
                      os.sep + 'help' + os.sep + 'index']

    data = []
    included_files = []
    error_files = []
    excluded_files = []
    for file_name in full_sitemap_list:
        # logger.info(f"Processing file {file_name}")
        try:
            loader = TextLoader(file_name)
            # ignore url's with /tag/ or /category/ as they do not contain relevant info.
            if any(exclusion in file_name for exclusion in exclusion_list):
                # logger.info(f"...exclusion found, not ingesting {file_name}")
                excluded_files.append(file_name)
                continue
            document = loader.load()
            # note h5 and h6 tags for our website contain a lot of irrelevant metadata
            doc_transformed = bs_transformer.transform_documents(
                document, tags_to_extract=[
                    "p", "article", "title", "h1"], unwanted_tags=[
                    "h5", "h6"], remove_lines=True)
            body_text = doc_transformed[0]

            # first remove duplicate spaces, then remove duplicate '\n\n', then remove duplicate '\n \n '
            body_text.page_content = re.sub(r'(\n ){2,}', '\n', re.sub(r'\n+', '\n', re.sub(r' +', ' ', body_text.page_content)))

            # remove the local directory from the source object
            body_text.metadata['source'] = body_text.metadata['source'].replace(website_generated_path, source_website_url)

            if len(body_text.page_content) > 100:
                data.append(body_text)
                included_files.append(file_name)
            else:
                # logger.info(f"document too small, not adding: {body_text.page_content}\n")
                excluded_files.append(file_name)
        except Exception as e:
            # logger.error(f"...unable to process file: {str(e)}")
            error_files.append(file_name)

    logger.info(f"==> Returning {len(included_files)} files; {len(error_files)} gave errors + {len(excluded_files)} files were skipped")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size/5)
    texts = text_splitter.split_documents(data)
    return texts

def remove_and_recreate(dir_path):
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"...removed directory {dir_path} and its contents.")
        os.makedirs(dir_path)
        logger.info(f"...directory {dir_path} (re)created.")
    except OSError as e:
        logger.error(f"Error: {e.strerror}")

def clone_and_generate(website_repo, destination_path, source_path):
    """
    Purpose:
        Retrieve the Hugo based website and generate the static html files.
    Args:
        website_repo: github repo containing the Hugo based website
        destination_path: path to directory containing generated html files
        source_path: path to directory containing the checked out github repo
    Returns:

    """

    logger.info(f"About to generate website: {website_repo}")
    remove_and_recreate(source_path)
    remove_and_recreate(destination_path)
    branch = "main"
    os.chdir(source_path)
    git_switch_command = ['git', 'switch', branch]
    git_directory = os.path.join(source_path, '.git')
    # Check if the repository already exists in the source_path
    if os.path.exists(git_directory):
        # Repository exists, perform a git pull to update it
        logger.info(f"...git directory exists, pulling in {os.getcwd()}")
        git_pull_command = ['git', 'pull', 'origin', branch]  # Modify branch name as needed
        result_pull = subprocess.run(git_pull_command, capture_output=True, text=True)
        if (result_pull.returncode != 0):
            logger.error(f"Unable to pull {website_repo} repository: {result_pull.stderr}")
        result_switch = subprocess.run(git_switch_command, cwd=source_path, capture_output=True, text=True)
        if (result_switch.returncode != 0):
            logger.error(f"Unable to switch {website_repo} repository: {result_switch.stderr}")
    else:
        logger.info(f"...git directory does not exist, cloning in {os.getcwd()}")  # Repository doesn't exist, perform a git clone
        clone_command = ['git', 'clone', "https://" + github_user + ":" + github_pat + "@" + website_repo, source_path]
        result_clone = subprocess.run(clone_command, capture_output=True, text=True)
        if (result_clone.returncode != 0):
            raise Exception(f"Unable to clone {website_repo} repository: {result_clone.stderr}")
        result_switch = subprocess.run(git_switch_command, cwd=source_path, capture_output=True, text=True)
        if (result_switch.returncode != 0):
            raise Exception(f"Unable to switch {website_repo} repository: {result_switch.stderr}")
        logger.info(f"git cloned + switch completed")

    os.chdir(source_path)
    logger.info(f"...cloned/updated, moved to directory: {os.getcwd()}")

    env = os.environ.copy()
    additional_path_go = '/usr/local/go/bin'
    additional_path_usr = '/usr/local'
    env["PATH"] = additional_path_go + os.pathsep + additional_path_usr + os.pathsep + env["PATH"]
    hugo_command = ['hugo', '--gc', '-b', '/', '-d', destination_path]
    logger.info(f"hugo command: {hugo_command}")
    result_hugo = subprocess.run(hugo_command, env=env, capture_output=True, text=True)
    if (result_hugo.returncode != 0):
        raise Exception(f"Unable to generate website using hugo command: '{hugo_command}': {result_hugo.stderr}")
    logger.info(f"hugo result: {result_hugo.stdout}")

    sitemap_file = create_sitemap_filepath(destination_path)
    if not os.path.exists(sitemap_file):
        raise Exception(f"Unable to generate website in {destination_path}: sitemap.xml not found: {sitemap_file}")


def ingest(source_url, website_repo, destination_path, source_path, source_url2, website_repo2, destination_path2, source_path2):
    clone_and_generate(website_repo, destination_path, source_path)
    clone_and_generate(website_repo2, destination_path2, source_path2)
    create_vector_db(source_url, source_url2)
    logger.info(f"Ingest successful")

def create_vector_db(source_website_url, source_website_url2) -> None:
    """
    Purpose:
        ingest the transformed website contents into a vector database in presized chunks.
    Args:
        source_website_url: full url of source website, used to return the proper link for the source documents.
    Returns:
        N/A
    """

    # open file to check output
    f = open(local_path+"/ingestion_output.txt", "w")

    # read and parse the files
    # local_source_path, source_website_url, website_generated_path
    texts = read_and_parse_html(website_source_path, source_website_url, website_generated_path)
    texts += read_and_parse_html(website_source_path2, source_website_url2, website_generated_path2)

    # Save embeddings to vectordb
    with get_openai_callback() as cb:
        embed_text(texts, vectordb_path)
    logger.info(f"\nEmbedding costs: {cb.total_cost}")
    stringified_texts = str(texts)
    f.write(stringified_texts)
    f.close()


# only execute if this is the main program run (so not imported)
if __name__ == "__main__":
    ingest(config['source_website'], config['website_repo'], website_generated_path, website_source_path,
           config['source_website2'], config['website_repo2'], website_generated_path2, website_source_path2)
