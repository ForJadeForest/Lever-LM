import json
import os
from typing import List

from datasets import Dataset
from loguru import logger


class VLGenInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}
    origin_image_dict = {}

    def __init__(
        self,
        num: int,
    ) -> None:
        self.num = num
        self.output_dict = {}
        self.prediction_dict = {}
        self.results_dict = {}
        self.other_meta_info_dict = {}
        self.idx_list = []
        self.origin_prompt_dict = {}

    def creat_index(self, test_ds: Dataset):
        self.idx_list = [i for i in range(len(test_ds))]

    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        self.results_dict = {
            str(idx): {
                "output": self.output_dict[str(idx)][0],
                "pure_output": self.output_dict[str(idx)][1],
                "prediction": self.prediction_dict[str(idx)],
            }
            for idx in self.idx_list
        }
        for field in self.other_meta_info_dict:
            for idx in self.idx_list:
                if field in self.results_dict[str(idx)]:
                    logger.warning(
                        "the other meta info field name has duplicate! Please check for avoiding to losing info"
                    )
                    continue
                self.results_dict[str(idx)][field] = self.other_meta_info_dict[field][
                    str(idx)
                ]
        save_path = f"{output_json_filepath}/{output_json_filename}.json"
        if not os.path.exists(output_json_filepath):
            os.makedirs(output_json_filepath)
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
            json_file.close()

    def save_prediction_and_output(self, prediction, output, origin_prompt, idx):
        self.prediction_dict[str(idx)] = prediction
        self.output_dict[str(idx)] = output
        self.origin_prompt_dict[str(idx)] = origin_prompt

    def save_origin_info(self, meta_field: str, test_ds: Dataset):
        meta_dict = {}
        meta_list = test_ds[meta_field]
        for idx, m_d in enumerate(meta_list):
            meta_dict[str(idx)] = m_d
        self.other_meta_info_dict[meta_field] = meta_dict


class PPLInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}
        self.other_meta_info_dict = {}

    def save_origin_info(self, meta_field: str, test_ds: Dataset):
        meta_dict = {}
        meta_list = test_ds[meta_field]
        for idx, m_d in enumerate(meta_list):
            meta_dict[str(idx)] = m_d
        self.other_meta_info_dict[meta_field] = meta_dict

    def creat_index(self, test_ds: Dataset):
        self.idx_list = [i for i in range(len(test_ds))]

    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        if not os.path.exists(output_json_filepath):
            os.makedirs(output_json_filepath)
        with open(
            f"{output_json_filepath}/{output_json_filename}.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
            json_file.close()

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]["in-context examples"] = example

    def save_predictions(self, predictions):
        for idx, prediction in enumerate(predictions):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]["prediction"] = prediction

    def save_prompt_and_ppl(self, label, input, prompt, ppl, idx):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if "label: " + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]["label: " + str(label)] = {}
        self.results_dict[str(idx)]["label: " + str(label)]["testing input"] = input
        self.results_dict[str(idx)]["label: " + str(label)]["prompt"] = prompt
        self.results_dict[str(idx)]["label: " + str(label)]["PPL"] = ppl
