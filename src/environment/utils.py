from dotenv import load_dotenv
import os


load_dotenv()

RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "25"))

# Auxiliary directories for the agent
DATA_DIR = os.getenv("DATA_DIR", "./.data_cache/")
PLOTS_DIR = os.getenv("PLOTS_DIR", "./plots/")

# Ollama configurations
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_MODEL_TEMPERATURE = float(os.getenv("OLLAMA_MODEL_TEMPERATURE", 0.7))

# Model Response Configurations
MODEL_RESPONSE_TIMEOUT = float(os.getenv("MODEL_RESPONSE_TIMEOUT", 120))

# Simulated database schema for the various (fictitious) tables
DATABASE_TABLES = {
    "customers": {
        "description": "Customer demographic and behavioral data",
        "columns": ["age (x)", "income (y)"],
        "cluster_characteristics": "Customer segmentation with distinct demographic and financial patterns",
    },
    "transactions": {
        "description": "Transaction records with amounts and frequencies",
        "columns": ["amount (x)", "frequency (y)"],
        "cluster_characteristics": "Transaction patterns with frequency variations",
    },
    "products": {
        "description": "Product performance and market positioning data",
        "columns": ["price (x)", "sales_volume (y)"],
        "cluster_characteristics": "Product categories with performance clusters",
    },
    "sensors": {
        "description": "IoT sensor readings from manufacturing equipment",
        "columns": ["temperature (x)", "pressure (y)"],
        "cluster_characteristics": "Equipment states with operational patterns",
    },
    "marketing": {
        "description": "Marketing campaign performance metrics",
        "columns": ["reach (x)", "engagement (y)"],
        "cluster_characteristics": "Campaign effectiveness with audience segments",
    },
}
