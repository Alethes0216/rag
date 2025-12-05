# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
from collections import defaultdict

class ReCallRewardManagerWithSave():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_path = save_path

    def __call__(self, data: DataProto, return_dict=False, curr_save_path=None):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        if save_path is not None:
            save_file = open(save_path, 'a')

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            
            # 后对齐，前面是0的去掉
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            score = self.compute_score(
                data_source=data_source,
                tokenizer=self.tokenizer,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            if isinstance(score, tuple):
                # score, reason, detail = score
                score, reason = score
            else:
                reason = ''
            reward_tensor[i, valid_response_length - 1] = score

            if save_path is not None:
                save_json_line = {
                    'data_source': data_source,
                    'sequences_str': sequences_str,
                    'ground_truth': ground_truth,
                    'score': score,
                    #'detail': detail,
                    'reason': reason
                }
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"data_source: \n{data_source}")
                print(f"sequences_str: \n{sequences_str}")
                print(f"ground_truth: \n{ground_truth}")
                print(f"score: \n{score}")  
                #print("detail:\n", detail)
                print(f"reason: \n{reason}")
                print('-' * 20)

        if save_path is not None:
            save_file.close()

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                #"reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor



if __name__ == "__main__":

    import os
    import json
    import pickle
    from omegaconf import DictConfig, OmegaConf

    def get_custom_reward_fn(config):
        import importlib.util, sys
        reward_fn_config = config.get("custom_reward_function") or {}
        file_path = reward_fn_config.get("path")
        if not file_path:
            return None

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}")

        function_name = reward_fn_config.get("name")
        if not hasattr(module, function_name):
            raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

        print(f"using customized reward function '{function_name}' from '{file_path}'")
        raw_fn = getattr(module, function_name)

        reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

        def wrapped_fn(*args, **kwargs):
            return raw_fn(*args, **kwargs, **reward_kwargs)

        return wrapped_fn


    with open("config_debug.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    config = OmegaConf.create(config)

    print("Loaded config:", config)

    os.makedirs(config["trainer"]["rollout_save_path"], exist_ok=True)

    pkl_path = "/root/recall/batch_debug_step_7.pkl"  # 之前保存的 batch 文件

    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # ---------------- 加载 batch ----------------
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    print(f"Loaded batch from {pkl_path}")

    # ---------------- 初始化 reward_fn ----------------
    compute_score = get_custom_reward_fn(config)

    reward_manager_cls = ReCallRewardManagerWithSave

    reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=0,      # 不打印 sample
        compute_score=compute_score
    )

    # ---------------- 运行 reward_fn ----------------
    save_path = os.path.join(config["trainer"]["rollout_save_path"], "train_debug.jsonl")
    reward_result = reward_fn(batch, return_dict=True, curr_save_path=save_path)

    reward_tensor = reward_result["reward_tensor"]
    print("Reward tensor shape:", reward_tensor.shape)
    print("Reward tensor:", reward_tensor)
    print(f"Results saved to {save_path}")