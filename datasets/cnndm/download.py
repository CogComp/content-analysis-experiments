"""Downloads the zip files with the Liu (2019) and See (2017) model outputs"""
import argparse
import os
from google_drive_downloader import GoogleDriveDownloader


def _download_file_from_google_drive(file_id: str, file_path: str, force: bool = False) -> None:
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if os.path.exists(file_path) and not force:
        print(f'Skipping downloading file {file_id}')
        return

    print(f'Downloading file {file_id} to {file_path}')
    GoogleDriveDownloader.download_file_from_google_drive(file_id, file_path, overwrite=True)


def main(args):
    _download_file_from_google_drive('1kYA384UEAQkvmZ-yWZAfxw7htCbCwFzC', args.liu_output_zip)
    _download_file_from_google_drive('0B7pQmm-OfDv7MEtMVU5sOHc5LTg', args.see_output_zip)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--liu-output-zip', required=True)
    argp.add_argument('--see-output-zip', required=True)
    args = argp.parse_args()
    main(args)