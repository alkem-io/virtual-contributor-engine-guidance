import subprocess
import re
import os
from bs4 import BeautifulSoup
from gitdb.db.pack import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import env
from git import Repo
import xml.etree.ElementTree as ET

from logger import setup_logger

logger = setup_logger(__name__)


def embed_text(documents: list[Document]):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=env.embeddings_model_name,
        chunk_size=1,
    )

    logger.info(f"Generating embeddings.")
    new_data = FAISS.from_documents(documents, embeddings)
    logger.info(f"Embeddings generated.")
    db_exists = (
        os.path.exists(env.vector_db_path) and len(os.listdir(env.vector_db_path)) > 0
    )
    if db_exists:
        logger.info(f"Local db exists. Load and add new data.")
        vector_store = FAISS.load_local(
            env.vector_db_path, embeddings, allow_dangerous_deserialization=True
        )
        vector_store.merge_from(new_data)
    else:
        logger.info(f"Local db does not exits. Initialise")
        vector_store = new_data

    vector_store.save_local(env.vector_db_path)
    logger.info(f"Local db updated.")


async def ensure_cloned(repo_name: str, path: str):
    if os.path.exists(path):
        logger.info(f"Repository {repo_name} already cloned.")
        repo = Repo.init(path)
    else:
        logger.info(f"Closning repository: {repo_name}")
        repo = Repo.clone_from(repo_name, path, branch="main")

    repo.heads.main.checkout()
    repo.remotes.origin.pull()
    logger.info("`main` branch checked out and latest changes pulled.")


def extract_urls_from_sitemap(path: str):
    logger.info("Generating sitemap URLs.")
    sitemap_file = os.path.join(path, "sitemap.xml")

    # Parse the XML directly from the file
    tree = ET.parse(sitemap_file)
    root = tree.getroot()

    # List to store the complete URLs of the webpages to be retrieved
    urls = []
    # Extract the URLs from the sitemap
    for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        # replace the / with the os separator
        if elem.text:
            url_path = elem.text.replace("/", os.sep)
            complete_url = path + url_path + "index.html"
            urls.append(complete_url)

    logger.info(f"{len(urls)} urls generated.")
    return urls


async def ensure_built(path: str):
    logger.info(f"Building site: {path}")
    site_hugo = ["hugo", "-s", path, "-d", "build"]
    subprocess.run(
        site_hugo,
        capture_output=True,
        text=True,
    )
    logger.info(f"Site {path} built.")


async def ensure_embedded(path: str, url: str):
    logger.info(f"Embedding {path}")
    files = extract_urls_from_sitemap(path)
    exclusion_list = [
        os.sep + "tag" + os.sep,
        os.sep + "category" + os.sep,
        os.sep + "help" + os.sep + "index",
    ]

    documents = []
    included_files = []
    error_files = []
    excluded_files = []
    tags = ["p", "article", "title", "h1"]
    logger.info(f"Files to embed: {len(files)}")
    logger.debug(f"Files to embed: {files}")
    for file_name in files:
        # logger.info(f"Processing file {file_name}")
        try:
            loader = TextLoader(file_name)
            # ignore url's with /tag/ or /category/ as they do not contain relevant info.
            if any(exclusion in file_name for exclusion in exclusion_list):
                # logger.info(f"...exclusion found, not ingesting {file_name}")
                excluded_files.append(file_name)
                continue

            [document] = loader.load()
            splitted = []
            document.page_content = re.sub(r"\n \n|\n", "", document.page_content)
            soup = BeautifulSoup(document.page_content, "html.parser")
            for tag in tags:
                matches = soup.find_all(tag)
                for match in matches:
                    splitted += match.get_text()
            document.page_content = "".join(splitted)
            # remove the local directory from the source object
            document.metadata["source"] = document.metadata["source"].replace(path, url)

            if len(document.page_content) > 100:
                logger.debug(
                    f"Document {document.metadata['source']} added for embedding."
                )
                documents.append(document)
                included_files.append(file_name)
            else:
                logger.debug(
                    f"Document too small, not adding: {document.metadata['source']}"
                )
                excluded_files.append(file_name)
        except Exception as e:
            logger.error(f"...unable to process file: {str(e)}")
            error_files.append(file_name)

    logger.info(f"{len(documents)} files added for embedidng")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=env.chunk_size, chunk_overlap=env.chunk_size / 5
    )
    splitted = text_splitter.split_documents(documents)
    logger.info(f"Documents splitted into {len(splitted)} chunks.")
    embed_text(splitted)
    return splitted


async def ingest_site(repo: str, dir: str, url: str):
    path = os.path.join(env.repos_path, dir)
    await ensure_cloned(repo, path)
    await ensure_built(path)
    await ensure_embedded(os.path.join(path, "build"), url)


def reset_db():
    logger.info("Resetting vector db.")
    files = glob.glob(os.path.join(env.vector_db_path, "*"))
    logger.debug(f"Files are {files}")
    for file in files:
        os.remove(file)
    logger.debug("Vector db reset.")


async def ensure_ingested(reset: bool = False):
    if reset:
        reset_db()
    logger.info("Ingestion started...")
    await ingest_site(env.welcome_site_repo, "welcome", env.welcome_site_url)
    await ingest_site(env.site_repo, "site", env.site_url)
    logger.info("Ingestion completed.")
