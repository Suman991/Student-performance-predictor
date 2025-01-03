import sys
import traceback
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logger import logger  # Import the logger instance

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if (exc_tb is not None):
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error: {error} at line {line_number} in {file_name}"
        return error_message
    else:
        return "No traceback available"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)  # Log the error message


# Example usage
if __name__ == "__main__":
    try:
    # Code that may raise an exception
        1 / 0
    except Exception as e:
        raise CustomException("Divide by Zero Error", sys)
