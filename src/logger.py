import logging
import os
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d__%H-%M-%S")}.log'
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#logging instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__=="__main__":
    # Example log message
    logger.info("Logging is configured.")