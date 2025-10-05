import base64
import logging
import boto3, os
from dotenv import load_dotenv
import fitz
from groq import Groq
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
import voyageai
import tempfile
from langchain_core.documents import Document
import hashlib
from sentence_transformers import SentenceTransformer

load_dotenv(".env")

b2_endpoint = os.getenv("B2_ENDPOINT_URL")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
b2_access_key = os.getenv("AWS_ACCESS_KEY_ID")
b2_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

def __get_b2_resource():
    b2_resource = boto3.resource(service_name='s3',
                        endpoint_url=b2_endpoint,     
                        aws_access_key_id=b2_access_key,
                        aws_secret_access_key=b2_secret_key)
    return b2_resource

def __upload_file_to_b2(b2_resource, local_file_path, bucket_name, object_name=None):
    """
    Upload a file to a B2 bucket
    
    :param b2_resource: Boto3 resource for B2
    :param local_file_path: Path to local file to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name (if None, uses basename of local_file_path)
    :return: True if file was uploaded, else False
    """
    # If object_name not specified, use file name from local_file_path
    if object_name is None:
        object_name = os.path.basename(local_file_path)
        
    try:
        # Get the bucket and upload the file
        bucket = b2_resource.Bucket(bucket_name)
        bucket.upload_file(local_file_path, object_name)
        logging.info(f"Successfully uploaded {local_file_path} to {bucket_name}/{object_name}")
        return True
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return False

def __download_file_from_b2(b2_resource, bucket_name, object_name, local_file_path):
    """
    Download a file from a B2 bucket
    
    :param b2_resource: Boto3 resource for B2
    :param bucket_name: Bucket to download from
    :param object_name: S3 object name to download
    :param local_file_path: Path where to save the downloaded file
    :return: True if file was downloaded, else False
    """
    try:
        # Get the bucket and download the file
        bucket = b2_resource.Bucket(bucket_name)
        bucket.download_file(object_name, local_file_path)
        logging.info(f"Successfully downloaded {bucket_name}/{object_name} to {local_file_path}")
        return True
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        return False
    
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def describe_image_with_llm(image_path):
    """
    Describe the image using Groq Vision API (LLaMA-4 Scout).
    """
    try:
        base64_image = encode_image(image_path)
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Image description failed: {e}")
        return f"Failed to describe image: {e}"
    
def __extract_text_and_images(downloaded_file_path, extract_images):
    print("here")
    documents = []
    # Step 1: Extract text content
    loader = PyMuPDFLoader(downloaded_file_path)
    doc_load = loader.load()
    documents.extend(doc_load)
    # doc_content = [doc.page_content for doc in doc_load]

    # Step 2: Extract and save images using fitz
    if extract_images:
        image_descriptions = []
        temp_image_dir = tempfile.mkdtemp(prefix="pdf_imgs_")
        
        pdf = fitz.open(downloaded_file_path)

        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"

                image_path = os.path.join(temp_image_dir, image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Step 3: Generate description using LLM
                description = describe_image_with_llm(image_path)

                image_descriptions.append({
                    "image_path": image_path,
                    "description": description
                })

                documents.append(
                    Document(
                        page_content=description,
                        metadata={
                            "type": "image_description",
                            "image_filename": image_filename,
                            "page": page_num + 1,
                            "source": downloaded_file_path
                        }
                    )
                )

    return {
        "text": documents,
        "images": image_descriptions if extract_images else []
    }


def __chunk_and_embed(documents, file_id, database_manager):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # custom_id = hashlib.md5(file_path.encode()).hexdigest()

    texts = [doc.page_content for doc in chunks]
    print("texts")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc.page_content for doc in chunks]
    
    # Generate embeddings locally
    embeddings = model.encode(texts, convert_to_tensor=False)
    embeddings = [emb.tolist() for emb in embeddings]
    print("embeddings done")
    database_manager.save_chunk_embeddings(chunks, embeddings, file_id=file_id)
    