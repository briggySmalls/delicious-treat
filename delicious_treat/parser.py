"""Parser for WhatsApp messages"""
import re
from datetime import datetime

import pandas as pd
import numpy as np


LINE_REGEX = r"(?P<datetime>\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2}) - (?P<name>.*?): (?P<message>.*)$"


def parse(text):
    # Compile the regex for matching a line
    regex = re.compile(LINE_REGEX)
    # Process the text
    data = []
    for line in text.split('\n'):
        # Match the line
        match = regex.match(line)
        if match:
            timestamp = datetime.strptime(match.group('datetime'), '%d/%m/%Y, %H:%M')
            data.append([timestamp, match.group('name'), match.group('message')])
    return pd.DataFrame(np.array(data), columns=['time', 'name', 'message'])
