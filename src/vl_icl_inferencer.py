from functools import partial
from typing import Optional

import datasets
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.lvlms.base_interface import BaseInterface
from src.utils import VLGenInferencerOutputHandler


class VLICLInferecer:
    def __init__(
        self,
        interface: BaseInterface,
        train_ds,
        test_ds,
        generation_kwargs,
        image_field,
        output_field,
        other_save_field=None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 8,
        num_proc: Optional[int] = 12,
        preprocessor_bs: Optional[int] = 100,
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
    ) -> None:
        self.interface = interface
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.generation_kwargs = generation_kwargs
        self.image_field = image_field
        self.other_save_field = other_save_field
        self.num_workers = num_workers
        self.num_proc = num_proc
        self.preprocessor_bs = preprocessor_bs
        self.batch_size = batch_size
        self.output_field = output_field
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename

    def inference(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        ice_idx_list,
    ):
        num = len(test_ds)
        output_handler = VLGenInferencerOutputHandler(num)
        index = 0

        test_ds = test_ds.add_column('ice_idx', ice_idx_list)

        def construct_prompts(
            examples,
        ):
            ice_idx = [i for i in examples['ice_idx']]
            prompts = []
            for i, e in enumerate(examples):
                ice_sample_list = [train_ds[idx] for idx in ice_idx[i]]
                data_sample_list = ice_sample_list + [e]
                prompt = self.interface.transfer_prompts(
                    data_sample_list, is_last_for_generation=True
                )
                prompts.append(prompt)
            return prompts

        test_ds = test_ds.map(
            construct_prompts,
            with_indices=True,
            batched=True,
            batch_size=self.preprocessor_bs,
        )
        test_ds = test_ds.cast_column(self.image_field, datasets.Image(decode=True))

        prepare_input_fn = partial(
            self.interface.prepare_input,
            add_eos_token=False,
            is_last_for_generation=True,
        )
        test_ds.set_transform(
            prepare_input_fn,
        )

        dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        output_handler.save_orgin_prompts(test_ds['prompt'])
        output_handler.save_origin_info('ice_idx', test_ds)
        for fields in self.other_save_field:
            output_handler.save_origin_info(fields, test_ds)
            # 4. Inference for prompts in each batch

        logger.info("Starting inference process...")
        for data in tqdm(dataloader, disable=not self.is_main_process, ncols=100):
            # 5-1. Inference with local model
            with self.autocast_context:
                prompt_len = int(data['attention_mask'].shape[1])
                outputs = self.model.generate(
                    **data,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_kwargs,
                )
                outputs = outputs.tolist()
                complete_output = self.tokenizer.batch_decode(
                    outputs[:], skip_special_tokens=False
                )
                generated = self.tokenizer.batch_decode(
                    [output[prompt_len:] for output in outputs],
                    skip_special_tokens=True,
                )

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1

        # 6. Output
        output_handler.subprocess_write_to_json(
            self.output_json_filepath, self.output_json_filename
        )

        output_handler.merge_to_main_process(
            self.output_json_filepath, self.output_json_filename
        )
        output_handler.write_to_json(
            self.output_json_filepath, self.output_json_filename
        )

        return output_handler.results_dict
