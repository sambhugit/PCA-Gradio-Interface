from google_drive_downloader import GoogleDriveDownloader as gdd
import os
current_dir = os.path.dirname(os.path.realpath('Gradio_Func_File.py'))

gdd.download_file_from_google_drive(file_id='1-u5Wd0JJ9JA2YQ0u-G-DcH8G7ExgveYX',
                                    dest_path= os.path.join(current_dir,'sentiment_weight_file.hdf5'),
                                    unzip=False)
