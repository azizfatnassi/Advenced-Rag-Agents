

import sys

from loguru import logger

def setup_logging():

    logger.remove()    # remove the predifined loguru handler(not json form,human formatted )
    logger.add(
        sys.stdout,
        serialize=True, 
        level="INFO",
        backtrace=False, #dont dump full traces into every log
        diagnose=False,  #avoid leaking variable values in logs (for security)
    ) 

    return logger 
