from src.custom_exception import CustomException
from src.logger import get_logger
import sys


logger = get_logger(__name__)

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        logger.error(e)
        raise CustomException("Test Exception", sys)


if __name__ == "__main__":
    divide(1, 0)


