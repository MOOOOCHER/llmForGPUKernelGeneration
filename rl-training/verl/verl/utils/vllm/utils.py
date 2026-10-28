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


import os

import torch
from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.third_party.vllm import get_version


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        if getattr(LRUCacheWorkerLoRAManager, "_verl_tensor_hijack_installed", False):
            return
        original_add_adapter = LRUCacheWorkerLoRAManager.add_adapter

        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        embeddings=None,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                if (
                    isinstance(lora_request, TensorLoRARequest)
                    and os.getenv("LORA_DIAGNOSTICS", "0").lower() in {"1", "true", "yes", "on"}
                ):
                    stats = {
                        "modules": len(lora.loras),
                        "a_numel": 0,
                        "b_numel": 0,
                        "a_nonzero": 0,
                        "b_nonzero": 0,
                        "a_max_abs": 0.0,
                        "b_max_abs": 0.0,
                    }

                    def iter_tensors(value):
                        if isinstance(value, torch.Tensor):
                            yield value
                        elif isinstance(value, dict):
                            for item in value.values():
                                yield from iter_tensors(item)
                        elif isinstance(value, (list, tuple)):
                            for item in value:
                                yield from iter_tensors(item)

                    for weights in lora.loras.values():
                        for prefix, value in (("a", weights.lora_a), ("b", weights.lora_b)):
                            for tensor in iter_tensors(value):
                                stats[f"{prefix}_numel"] += tensor.numel()
                                stats[f"{prefix}_nonzero"] += int(torch.count_nonzero(tensor).item())
                                if tensor.numel():
                                    stats[f"{prefix}_max_abs"] = max(
                                        stats[f"{prefix}_max_abs"], float(tensor.abs().max().item())
                                    )
                    print(
                        f"[LoRA][vllm-load-diagnostics] name={lora_request.lora_name} "
                        f"id={lora_request.lora_int_id} rank={lora.rank} stats={stats}",
                        flush=True,
                    )
            except Exception as e:
                raise e

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        def hijack_add_adapter(self, lora_request: LoRARequest) -> bool:
            """Add an adapter and optionally inspect its active GPU slot."""
            loaded = original_add_adapter(self, lora_request)
            if os.getenv("LORA_DIAGNOSTICS", "0").lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return loaded

            diagnosed = getattr(self, "_verl_diagnosed_active_loras", set())
            lora_id = lora_request.lora_int_id
            diagnostic_key = (lora_id, lora_request.lora_name)
            if diagnostic_key in diagnosed:
                return loaded

            manager = self._adapter_manager
            try:
                slot = manager.lora_index_to_id.index(lora_id)
            except ValueError:
                print(
                    f"[LoRA][vllm-active-diagnostics] name={lora_request.lora_name} "
                    f"id={lora_id} active_ids={manager.lora_index_to_id} "
                    "error=adapter_not_in_gpu_slot",
                    flush=True,
                )
                return loaded

            stats = {
                "modules": 0,
                "a_tensors": 0,
                "b_tensors": 0,
                "a_numel": 0,
                "b_numel": 0,
                "a_nonzero": 0,
                "b_nonzero": 0,
                "a_nonfinite": 0,
                "b_nonfinite": 0,
                "a_max_abs": 0.0,
                "b_max_abs": 0.0,
            }

            def update_stats(prefix, value):
                values = value if isinstance(value, (list, tuple)) else (value,)
                for tensor in values:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    active = tensor[slot]
                    stats[f"{prefix}_tensors"] += 1
                    stats[f"{prefix}_numel"] += active.numel()
                    stats[f"{prefix}_nonzero"] += int(torch.count_nonzero(active).item())
                    stats[f"{prefix}_nonfinite"] += int(
                        torch.count_nonzero(~torch.isfinite(active)).item()
                    )
                    if active.numel():
                        finite = active[torch.isfinite(active)]
                        if finite.numel():
                            stats[f"{prefix}_max_abs"] = max(
                                stats[f"{prefix}_max_abs"],
                                float(finite.abs().max().item()),
                            )

            for module in manager.modules.values():
                stats["modules"] += 1
                update_stats("a", module.lora_a_stacked)
                update_stats("b", module.lora_b_stacked)

            print(
                f"[LoRA][vllm-active-diagnostics] name={lora_request.lora_name} "
                f"id={lora_id} slot={slot} active_ids={manager.lora_index_to_id} "
                f"stats={stats}",
                flush=True,
            )
            if (
                os.getenv("LORA_DIAGNOSTICS_EXPECT_ZERO", "0").lower()
                in {"1", "true", "yes", "on"}
                and stats["b_nonzero"] != 0
            ):
                raise RuntimeError(
                    f"Active vLLM LoRA B is not zero for "
                    f"{lora_request.lora_name!r}: {stats}"
                )

            diagnosed.add(diagnostic_key)
            self._verl_diagnosed_active_loras = diagnosed
            return loaded

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)
        do_hijack(LRUCacheWorkerLoRAManager, "add_adapter", hijack_add_adapter)
        LRUCacheWorkerLoRAManager._verl_tensor_hijack_installed = True


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
