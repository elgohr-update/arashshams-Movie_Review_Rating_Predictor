"""Download file from an URL to a local path

Usage:
  download_data.py <url> <download_path>
  download_data.py (-h | --help)

Options:
  <url>             URL to the file to download
  <download_path>   Path (including filename) where the downloaded file should be stored
  -h, --help        Display help
"""
import os
import urllib.request
from pathlib import Path

from docopt import docopt


def main(url, out_file):
    # Create parent directories recursively if not exist
    Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)

    # Download file from URL
    try:
        urllib.request.urlretrieve(url, out_file)
    except Exception as e:
        # While we know that catching all exceptions is not ideal, it's well-known that urllib have a
        # pretty bad support for exception handling, so this is the best thing we can do here.
        print(f'The following error occurs during file downloading: \n{str(e)}')
        exit(-1)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(f'Downloading {arguments["<url>"]} to {arguments["<download_path>"]}')
    main(arguments["<url>"], arguments["<download_path>"])
