from typing import Optional

import datasets
from torch.utils.data import DataLoader

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
        train_ds,
        test_ds,
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
                prompts.append(self.interface.construct_prompt(
                    ice_sample_list,
                    
                ))
            return prompts

        test_ds = test_ds.map(
            construct_prompts,
            with_indices=True,
            batched=True,
            batch_size=self.preprocessor_bs,
        )
        test_ds = test_ds.cast_column(self.image_field, datasets.Image(decode=True))

        prepare_input_fn = self.interface.prepare_input
        test_ds.set_transform(prepare_input_fn)

        dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
