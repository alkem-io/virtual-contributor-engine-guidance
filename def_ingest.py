import os
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import xml.etree.ElementTree as ET


from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import BSHTMLLoader
from bs4 import BeautifulSoup

import xml.etree.ElementTree as ET

# define local configuration parameters
local_path = os.getenv('AI_LOCAL_PATH')
website_source_path = local_path+"/website-source"
website_generated_path = local_path+"/website-generated"
vectordb_path = local_path+"/local_index"


def extract_urls_from_sitemap(sitemap_location='sitemap.xml', base_directory='./'):
    """
    Purpose:
        Read the sitemap.xml file and create a list of local html files to be read
    Args:
        sitemap_location: full path to sitemap.xml
        base_directory: path to directory containing local html files
    Returns:
        list of files to be retrieved
    """
    # Parse the XML directly from the file
    tree = ET.parse(sitemap_location)
    root = tree.getroot()

    # Extract the URLs from the sitemap
    to_be_retieved = [
        base_directory + elem.text + "index.html"
        for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    ]

    return to_be_retieved


def load_html_text(target_files):

    loader = BSHTMLLoader(target_files)
    data = loader.load()

    # Transform
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(data, tags_to_extract=["body"])

    return docs_transformed


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
    # Get all links from the sitemaps
    full_sitemap_list = extract_urls_from_sitemap(website_generated_path+"/sitemap.xml", website_generated_path)
    print(full_sitemap_list)

    data = []
    for file_name in full_sitemap_list:
        loader = BSHTMLLoader(file_name)
        document = loader.load()

        body_text = document[0]

        # first remove duplicate spaces, then remove duplicate '\n\n', then remove duplicate '\n \n '
        body_text.page_content = re.sub(r'(\n ){2,}', '\n', re.sub(r'\n+', '\n', re.sub(r' +', ' ', body_text.page_content)))

        # remove the local directory from the source object
        body_text.metadata['source'] = body_text.metadata['source'].replace(local_source_path, source_website_url)

        data.append(body_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

def clone_and_generate(website_repo, destination_path, source_path):
    # clone repo and generate files
    os.system(f"bash /app/generate-website.sh {website_repo} {destination_path} {source_path}")

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

    # Save embeddings to local_index
    embed_text(texts, vectordb_path)

    f.write(str(texts))
    f.close()


# only execute if this is the main program run (so not imported)
if __name__ == "__main__":
    mainapp(os.getenv('AI_SOURCE_WEBSITE'))
