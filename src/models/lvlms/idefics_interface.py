import torch
from loguru import logger
from transformers import BatchFeature, IdeficsForVisionText2Text, IdeficsProcessor

from .base_interface import BaseInterface


class IDEFICSInterface(BaseInterface):
    def __init__(
        self,
        hf_root,
        load_from_local,
        precision,
        device,
        prompt_template,
        column_token_map,
        icd_token,
        instruction,
        icd_join_char='\n',
    ):
        super().__init__(
            precision=precision,
            device=device,
            input_ids_filed_name='input_ids',
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            icd_token=icd_token,
            instruction=instruction,
            icd_join_char=icd_join_char,
        )
        self.processor = IdeficsProcessor.from_pretrained(
            hf_root, local_files_only=load_from_local, init_config=init_config
        )
        self.model = IdeficsForVisionText2Text.from_pretrained(
            hf_root, torch_dtype=self.data_type, local_files_only=load_from_local
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = 'left'
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id

        self.fake_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        self.image_prompt = self.fake_token + self.image_token + self.fake_token

    def add_image_token(self, text):
        return self.image_prompt + text

    def construct_prompt(
        self,
        text_list,
        add_join_token_end=False,
        add_eos_token=False,
        add_image_token=False,
    ):
        """return the prompt: <s><Instruction>[<IMAGE_TOKEN>]text1<icd_join_char> ... textn[<icd_join_char>][</s>]

        Args:
            text_list (_type_): _description_
            add_join_token_end (bool, optional): _description_. Defaults to False.
            add_eos_token (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        text_input = self.tokenizer.bos_token + self.instruction
        if not text_list:
            return text_input
        if add_image_token:
            text_list = [self.add_image_token(t.strip(' ')) for t in text_list]
        text_input += self.icd_join_char.join(text_list)
        if add_join_token_end:
            text_input += self.icd_join_char
        if add_eos_token:
            text_input += self.tokenizer.eos_token
        return text_input

    def prepare_input(
        self,
        batch_prompts,
        add_join_token_end=False,
        add_eos_token=False,
        debug=False,
        transform=None,
    ):
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"

        def image_tokens(last_was_image):
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token

        all_images = []
        all_raw_texts = []
        for sample in batch_prompts:
            # the model was trained on samples starting with <s>
            full_text = f"{self.tokenizer.bos_token}"

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            image_objects = []
            last_was_image = False

            for i, item in enumerate(sample):
                item_is_img = self.is_img(item)
                if item_is_img is None:
                    item = item.strip(" ")
                    full_text += item
                    if i != len(sample) - 1 or add_join_token_end:
                        full_text += self.icd_join_char
                    last_was_image = False
                else:
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item_is_img)
                    last_was_image = True

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            if debug is True:
                print(f"{full_text=}")

            image_objects = self.image_processor(image_objects, transform=transform)

            all_raw_texts.append(full_text)
            all_images.append(image_objects)

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        at_least_one_image = sum(len(x) for x in all_images) > 0
        output_images = []
        text_tensor_input = self.tokenizer(
            all_raw_texts, padding=True, add_special_tokens=False, return_tensors='pt'
        )
        for text_tensor, images in zip(text_tensor_input['input_ids'], all_images):
            image_count = (text_tensor == self.image_token_id).sum()

            local_max_num_images = min(image_count, max_num_images)
            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(
                    max_num_images, *current_images.size()[1:]
                )
                padded_image_tensor[: current_images.size(0)] = current_images
            else:
                padded_image_tensor = torch.zeros(
                    max_num_images, *self.default_image_dims
                )

            output_images.append(padded_image_tensor)

        output_input_ids = text_tensor_input['input_ids']
        output_images = torch.stack(output_images)
        output_attention_masks = text_tensor_input['attention_mask']

        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(
                output_input_ids, self.tokenizer
            )
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, num_classes=max_num_images
            )
        else:
            # in full language mode we set the image mask to all-0s
            image_attention_mask = torch.zeros(
                output_input_ids.shape[0],
                output_input_ids.shape[1],
                1,
                dtype=torch.bool,
            )
        return BatchFeature(
            data={
                "input_ids": output_input_ids,
                "attention_mask": output_attention_masks,
                "pixel_values": output_images,
                "image_attention_mask": image_attention_mask,
            }
        ).to(self.device)


# copied from m4.training.packing
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer):
    image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx, token_id in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count

            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1

            if token_id == eod_token_id:
                seen_eod = True

    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            token_id = input_ids[batch_idx][idx]
            if token_id == image_token_id:
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                next_image_attention_mask[batch_idx][idx] = count

            if token_id == eod_token_id:
                seen_eod = True

            if seen_eod:
                next_image_attention_mask[batch_idx][idx] = -1

        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1

    return image_attention_mask, next_image_attention_mask


# copied from m4.training.packing
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    # This function converts: [-1, 0, 1] => [[0, 0], [1, 0], [0, 1]]

    # If any of images index are more than num_classes, set them to -1.
    # Words after the max number of images allowed have been seen don't attend on anything
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1

    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask
