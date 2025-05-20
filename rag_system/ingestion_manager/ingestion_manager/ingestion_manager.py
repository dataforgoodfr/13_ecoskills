import pathlib
import json
import logging
from datetime import datetime

class IngestionManager():

    def __init__(self, pdf_path):

        self.target_folder = pathlib.Path(pdf_path)
        self.json_file_path = self.target_folder / 'ingestion_report.json'

    def _init_folder_ingestion_report(self):

        if self.json_file_path not in self.target_folder.iterdir():
            init_dict = {'init':'ok',
                           'date_init': datetime.now().isoformat()}
            #write JSON files:
            with self.json_file_path.open("w", encoding="UTF-8") as json_file:
                json.dump(init_dict, json_file, indent=4)
                logging.info("Ingestion json report file initialized.")

        else:
            logging.info("Ingestion json report file is already initialized.")

    def _write_report(self, pdf_filename, ingestion_report):

        if self.json_file_path not in self.target_folder.iterdir():
            self._init_folder_ingestion_report()

        with self.json_file_path.open("r", encoding="UTF-8") as json_file:
            json_data = json.load(json_file)
            pdf_report = json_data.get('pdfs', {})

        try:
            pdf_report[pdf_filename] = ingestion_report
            json_data['pdfs'] = pdf_report
            with self.json_file_path.open("w", encoding="UTF-8") as json_file: 
                json.dump(json_data, json_file, indent=4)
                logging.info("Ingestion json updated.")

        except Exception as e:
            logging.warning(f"Ingestion report failed ! --- pdf_filename : {pdf_filename} - error : {e}")

    def _get_ingestion_report(self) -> dict:

        if self.json_file_path not in self.target_folder.iterdir():
            logging.warning(" No ingestion report json file found...")
            logging.warning(" Generation...")
            self._init_folder_ingestion_report()

        with self.json_file_path.open("r", encoding="UTF-8") as json_file:
            json_data = json.load(json_file)
            return json_data
                