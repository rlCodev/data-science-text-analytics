from dotenv import load_dotenv
import os
from pydantic import BaseSettings

# Load environment variables
class Settings(BaseSettings):
    elastic_host: str = os.getenv("APP_NAME")
    elastic_pw: str = os.getenv("ELASTIC_PW")
    elastic_user: str = os.getenv("ELASTIC_USER")
    class Config:
        env_prefix = ''
        env_file = "api/.env"
        env_file_encoding = 'utf-8'

settings = Settings()