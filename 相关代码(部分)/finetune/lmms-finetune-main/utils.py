import math
from typing import List, Dict, Optional

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length
import torch.nn.functional as F

class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化 ne_margin_loss 相关的参数
        self.ne_margin_loss = True
        self.ne_margin_loss_weight = 0.3
        self.ne_margin_threshold = 5
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )


    
    
    



    def compute_loss1(self, model, inputs, return_outputs=False):
        # 模型的前向传播
        print("This is in puts+++++++++++++++++=",inputs)
        outputs = model(**inputs)
        print("This is out puts+++++++++++++++++=",outputs)
        # 获取模型输出
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        batch_size = logits_per_text.size(0)
        logit_scale = model.logit_scale.exp()

        # 计算标准的损失
        per_image_loss = torch.nn.functional.cross_entropy(
            logits_per_image, torch.arange(len(logits_per_image), device=logits_per_image.device, dtype=torch.long)
        )
        per_text_loss = torch.nn.functional.cross_entropy(
            logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device, dtype=torch.long)
        )

        loss = (per_image_loss + per_text_loss) / 2

        # 计算 ne_margin_loss，如果启用了该选项
        if self.ne_margin_loss:
            # 计算负样本的相似度
            neg_text = torch.cat([logits_per_text.unsqueeze(1)] * batch_size, dim=1)
            ne_logits_per_text = torch.matmul(neg_text, image_embeds.unsqueeze(1).permute(0, 2, 1)).squeeze(-1) * logit_scale
            true_logits_per_text = logits_per_text.diagonal().unsqueeze(-1)
            batch_ne_logits_per_text = logits_per_text[~torch.eye(batch_size, dtype=bool)].reshape(batch_size, batch_size - 1)

            # 计算 ne_margin_loss
            ne_margin_loss = torch.nn.functional.relu(
                self.ne_margin_threshold - ne_logits_per_text +
                batch_ne_logits_per_text.max(dim=1)[0].unsqueeze(-1)
            )

            # 将 ne_margin_loss 加入到总损失中
            loss = loss + self.ne_margin_loss_weight * ne_margin_loss.mean()

        # 返回损失
        return (loss, outputs) if return_outputs else loss

def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)