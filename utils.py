import re

def clean_title(title):
    """Removes special characters from a movie title."""
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title