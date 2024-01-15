from typing import Optional

import torch
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from open_mmicl.lvlm_interface.base_interface import BaseInterface
from icd_lm.utils import VLGenInferencerOutputHandler


class VLICLInferecer:
    def __init__(
        self,
        interface: BaseInterface,
        train_ds,
        test_ds,
        generation_kwargs,
        other_save_field=None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 8,
        num_proc: Optional[int] = 12,
        preprocessor_bs: Optional[int] = 100,
        output_json_filepath: Optional[str] = "./vl_icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
    ) -> None:
        self.interface = interface
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.generation_kwargs = generation_kwargs
        self.other_save_field = other_save_field
        self.num_workers = num_workers
        self.num_proc = num_proc
        self.preprocessor_bs = preprocessor_bs
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename

    @torch.inference_mode()
    def inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename
        num = len(test_ds)
        output_handler = VLGenInferencerOutputHandler(num)
        index = 0

        test_ds = test_ds.add_column('ice_idx', ice_idx_list)

        output_handler.creat_index(test_ds)
        output_handler.save_origin_info('ice_idx', test_ds)
        for fields in self.other_save_field:
            output_handler.save_origin_info(fields, test_ds)

        def prepare_input_map(
            examples,
        ):
            ice_idx = [i for i in examples['ice_idx']]
            prompts = []
            num_example = len(ice_idx)
            sub_data_sample = [
                {k: v[i] for k, v in examples.items()} for i in range(num_example)
            ]
            batch_data_smaple_list = []
            for i, e in enumerate(sub_data_sample):
                ice_sample_list = [train_ds[idx] for idx in ice_idx[i]]
                data_sample_list = ice_sample_list + [e]
                batch_data_smaple_list.append(data_sample_list)
            prompts = self.interface.transfer_prompts(
                batch_data_smaple_list, is_last_for_generation=True
            )

            input_tensor_dict = self.interface.prepare_input(
                prompts, is_last_for_generation=True
            )
            return input_tensor_dict

        test_ds.set_transform(prepare_input_map)
        dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        # 4. Inference for prompts in each batch
        logger.info("Starting inference process...")

        for data in tqdm(dataloader, ncols=100):
            # 5-1. Inference with local model

            data = {k: v.to(self.interface.device) for k, v in data.items()}
            prompt_len = int(data['attention_mask'].shape[1])
            outputs = self.interface.generate(
                **data,
                eos_token_id=self.interface.tokenizer.eos_token_id,
                pad_token_id=self.interface.tokenizer.pad_token_id,
                **self.generation_kwargs,
            )
            outputs = outputs.tolist()
            complete_output = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=False
            )
            output_without_sp_token = self.interface.tokenizer.batch_decode(
                outputs[:], skip_special_tokens=True
            )
            generated = self.interface.tokenizer.batch_decode(
                [output[prompt_len:] for output in outputs],
                skip_special_tokens=True,
            )
            origin_prompt = self.interface.tokenizer.batch_decode(
                [output[:prompt_len] for output in outputs],
                skip_special_tokens=True,
            )

            # 5-3. Save current output
            for prediction, output, pure_output in zip(
                generated, complete_output, output_without_sp_token
            ):
                output_handler.save_prediction_and_output(
                    prediction, [output, pure_output], origin_prompt, index
                )
                index = index + 1

        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return output_handler.results_dict
