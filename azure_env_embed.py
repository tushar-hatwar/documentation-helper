from dotenv import load_dotenv, find_dotenv
import os
from pydantic import SecretStr
from langchain_openai import AzureOpenAIEmbeddings

# Load .env reliably from the script directory when cwd differs
env_path = find_dotenv()
if not env_path:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
loaded = load_dotenv(env_path)

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "")

embeddings = AzureOpenAIEmbeddings(
    api_key        = SecretStr(AZURE_API_KEY) if AZURE_API_KEY is not None else None,
    api_version    = "2024-08-01-preview",
    azure_endpoint = "https://ai-proxy.lab.epam.com",
    model          = AZURE_EMBEDDING_MODEL,
    check_embedding_ctx_length = False,
)