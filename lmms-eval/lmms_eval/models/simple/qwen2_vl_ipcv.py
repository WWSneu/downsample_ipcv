import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import pdb
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
#from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer
import sys
sys.path.append('~/ViT-Prunning-self/Qwen2-VL/')
from Qwen2VL_IPCV import Qwen2VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


from ..flops_kv_monitor import KVFlopsMeter
import time


def configure_Sparse(model, config):
    model.config.Sparse_config = config
    model.visual.config.Sparse_config = config
    
###################################################################
# Qwen2_VL_DR Model
# Parameter setting area.
scale_factor_default: int = 4  # Define scale_factor globally for the module
###################################################################

@register_model("qwen2_vl_ipcv")
class Qwen2_VL_IPCV(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        # max_pixels: int = 12845056,
        max_pixels: int = 2408448,
        min_pixels: int = 3136,
        # max_pixels: int = 602112, # video setting
        # min_pixels: int = 3136, # video setting
        # max_pixels: int = 16384*28*28, # default setting
        # min_pixels: int = 1280*28*28, # default setting
        max_num_frames: int = 32,

        attn_implementation="flash_attention_2",
        Sparse=True,
        pruned_layer=2,
        reduction_ratio=0.778,
        vit_Sparse=True,
        vit_pruned_layer=2,
        vit_reduction_ratio=0.778,
        image_token_start_index=0,
        image_token_length=0,
        max_num_trunction=0,
        pivot_image_token=4,
        pivot_text_token=4,
        random_choose = False,
        attn_scores_choose=False,
        diff_choose = False,
        pivot_sim_choose = False,

        vit_random_choose = False,
        vit_attn_scores_choose=False,
        vit_diff_choose = False,
        vit_pivot_sim_choose = False,

        torch_dtype="auto",

        AS_layer=3,
        Top_K=10,
        scale_factor: int = scale_factor_default,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        #print("max_pixels",max_pixels,min_pixels)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            # self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            #     pretrained,
            #     torch_dtype="auto",
            #     device_map=self.device_map,
            #     attn_implementation="flash_attention_2",
            # ).eval()
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            # self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map=self.device_map).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        #print("config_size",pretrained,self.processor.text_config.hidden_size, self.processor.text_config.vocab_size)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.scale_factor = scale_factor
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        # Inject scale_factor into the underlying model so it can be accessed in modeling code
        # Unwrap model to handle DDP/FSDP wrappers
        unwrapped_model = accelerator.unwrap_model(self._model)
        
        # 1. Bind to outer wrapper (ForConditionalGeneration)
        unwrapped_model.scale_factor = scale_factor
        
        # 2. Bind to inner model (Qwen2VLModel) where get_image_features is called
        if hasattr(unwrapped_model, "model"):
             unwrapped_model.model.scale_factor = scale_factor
        else:
             eval_logger.warning("Could not find inner 'model' attribute to inject scale_factor. Resampling inside model might fail.")
        Sparse_config = {
            "Sparse": Sparse,
            "pruned_layer": pruned_layer,
            "reduction_ratio": reduction_ratio,
            "vit_Sparse": vit_Sparse,
            "vit_pruned_layer": vit_pruned_layer,
            "vit_reduction_ratio": vit_reduction_ratio,

            "image_token_start_index": image_token_start_index,
            "image_token_length": image_token_length,
            "max_num_trunction": max_num_trunction,
            "pivot_image_token": pivot_image_token,
            "pivot_text_token": pivot_text_token,

            "AS_layer": AS_layer,
            "Top_K": Top_K
        }
        configure_Sparse(self._model, Sparse_config) # HACK

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def lanczos_downsample(self, image_inputs: List[Image.Image], scale_factor: float = scale_factor_default) -> List[Image.Image]:
        """
        对输入的 PIL Image 列表进行 Lanczos 下采样。
        
        参数:
            image_inputs: 包含 PIL.Image.Image 对象的列表
            scale_factor: 缩放倍率 (例如 0.5 表示长宽缩小为一半)。如果不传参，默认值为 0.5。
            
        返回:
            下采样后的 PIL Image 列表
        """
        downsampled_list = []
        
        for img in image_inputs:
            # 获取原始尺寸
            width, height = img.size

            eval_logger.info(f"Original image size in lanczos: {img.size}")
            # 计算新尺寸
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            eval_logger.info(f"Resized image size in lanczos before adjust: {(new_width, new_height)}")

            # 调整为 28 的倍数
            new_width = (new_width // 28) * 28
            new_height = (new_height // 28) * 28
            eval_logger.info(f"Resized image size in lanczos after adjust: {(new_width, new_height)}")
            # 确保至少为 28x28 (避免尺寸过小)
            new_width = max(28, new_width)
            new_height = max(28, new_height)
            
            # 使用 Lanczos 滤波器进行缩放
            # Image.Resampling.LANCZOS 是新版 Pillow 的写法，旧版可用 Image.LANCZOS
            resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            
            downsampled_list.append(resized_img)
        for img in downsampled_list:
            eval_logger.info(f"Downsampled image size: {img.size}")
            # pass  # 可以在这里添加日志或其他处理
        return downsampled_list

    def resize_to_multiple(self, image_inputs: List[Image.Image], scale_factor: float = scale_factor_default) -> List[Image.Image]:
        """
        Resize images to be multiples of 14/scale_factor.
        This ensures that downsampled images (by scale_factor) will be multiples of 14.
        """
        resized_list = []
        # base should be the target block size in high-res space
        base = 28.0 / scale_factor
        
        for img in image_inputs:
            width, height = img.size

            eval_logger.info(f"Original image size: {img.size}")
            # Round to nearest multiple of base
            new_width = int(round(width / base) * base)
            new_height = int(round(height / base) * base)

            eval_logger.info(f"Resized image size: {(new_width, new_height)}")
            # Ensure minimum size
            new_width = max(int(base), new_width)
            new_height = max(int(base), new_height)

            eval_logger.info(f"Resized image size after max: {(new_width, new_height)}")

            if new_width != width or new_height != height:
                resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
                resized_list.append(resized_img)
            else:
                resized_list.append(img)
            eval_logger.info(f"Final image size: {resized_list[-1].size}")
        return resized_list
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        

        # ---------------compute kv1-----------------------------
        compute_flops_kv = False
        if compute_flops_kv:
            meter = KVFlopsMeter(self.model)
            meter.start()
        # ---------------compute kv-----------------------------

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # ---------------compute time-------------
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_infer_time = 0.0
        # ---------------compute time-------------
        sample_num=0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            if None in visuals:
                visuals = []
            else:
                visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            # messages = []
            # processed_visuals = []
            # for i, context in enumerate(contexts):
            #     # print(f"\033[94mtext: {context}\033[0m")
            #     if "<image>" in context:
            #         context = context.replace("<image>", "")

            #     message = [{"role": "system", "content": "You are a helpful assistant."}]

            #     if len(visuals) > 0:
            #         visual = visuals[i] if i < len(visuals) else None
            #         if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
            #             vr = decord.VideoReader(visual)
            #             first_frame = vr[0].asnumpy()
            #             height, width = first_frame.shape[:2]
            #             # max_pixels = height * width
            #             message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
            #         elif isinstance(visual, Image.Image):  # Single image
            #             eval_logger.info(f"[Image] 原始尺寸: {visual.size}")
            #             # 2倍下采样
            #             w, h = visual.size
            #             visual = visual.resize((w // scale_factor_default, h // scale_factor_default), resample=Image.BICUBIC)
            #             eval_logger.info(f"[Image] 下采样后尺寸: {visual.size}")

            #             base64_image = visual.convert("RGB")
            #             buffer = BytesIO()
            #             base64_image.save(buffer, format="JPEG")
            #             base64_bytes = base64.b64encode(buffer.getvalue())
            #             base64_string = base64_bytes.decode("utf-8")
            #             message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
            #         elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
            #             image_content = []
            #             for v in visual:

            #                 eval_logger.info(f"[Image] 原始尺寸: {v.size}")
            #                 w, h = v.size
            #                 v = v.resize((w // scale_factor_default, h // scale_factor_default), resample=Image.BICUBIC)
            #                 eval_logger.info(f"[Image] 下采样后尺寸: {v.size}")

            #                 base64_image = v.convert("RGB")
            #                 buffer = BytesIO()
            #                 base64_image.save(buffer, format="JPEG")
            #                 base64_bytes = base64.b64encode(buffer.getvalue())
            #                 base64_string = base64_bytes.decode("utf-8")
            #                 image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
            #             message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            #         else:
            #             message.append({"role": "user", "content": [{"type": "text", "text": context}]})
            #     else:
            #         message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            #     messages.append(message)

            # texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            # # text1 = texts
            # # text2 = texts
            # # text3 = texts
            # image_inputs, video_inputs = process_vision_info(messages)
            messages_orig = []   # 原图版本
            messages_down = []   # 下采样版本
            processed_visuals = []

            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                # 两套 message 都要有 system
                msg_orig = [{"role": "system", "content": "You are a helpful assistant."}]
                msg_down = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None

                    # ---------- 视频：一般不下采样，两个 message 一样 ----------
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]

                        user_content_video = [
                            {"type": "video", "video": visual, "max_pixels": self.max_pixels},
                            {"type": "text", "text": context},
                        ]
                        msg_orig.append({"role": "user", "content": user_content_video})
                        msg_down.append({"role": "user", "content": user_content_video})

                    # ---------- 单张图片 ----------
                    elif isinstance(visual, Image.Image):
                        # 原图版本
                        eval_logger.info(f"[Image-ORIG] 原始尺寸: {visual.size}")
                        base64_image_orig = visual.convert("RGB")
                        buffer_orig = BytesIO()
                        base64_image_orig.save(buffer_orig, format="JPEG")
                        base64_bytes_orig = base64.b64encode(buffer_orig.getvalue())
                        base64_string_orig = base64_bytes_orig.decode("utf-8")

                        msg_orig.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"data:image/jpeg;base64,{base64_string_orig}"},
                                {"type": "text", "text": context},
                            ],
                        })

                        # 下采样版本
                        v_ds = visual  # PIL 的 resize 会返回新对象，这里只是命名
                        eval_logger.info(f"[Image-DOWN] 原始尺寸: {v_ds.size}")
                        w, h = v_ds.size
                        v_ds = v_ds.resize(
                            (w // scale_factor_default, h // scale_factor_default),
                            resample=Image.BICUBIC,
                        )
                        eval_logger.info(f"[Image-DOWN] 下采样后尺寸: {v_ds.size}")

                        base64_image_down = v_ds.convert("RGB")
                        buffer_down = BytesIO()
                        base64_image_down.save(buffer_down, format="JPEG")
                        base64_bytes_down = base64.b64encode(buffer_down.getvalue())
                        base64_string_down = base64_bytes_down.decode("utf-8")

                        msg_down.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"data:image/jpeg;base64,{base64_string_down}"},
                                {"type": "text", "text": context},
                            ],
                        })

                    # ---------- 多张图片 ----------
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                        image_content_orig = []
                        image_content_down = []

                        for v in visual:
                            # 原图版本
                            eval_logger.info(f"[Multi-Image-ORIG] 原始尺寸: {v.size}")
                            base64_image_orig = v.convert("RGB")
                            buffer_orig = BytesIO()
                            base64_image_orig.save(buffer_orig, format="JPEG")
                            base64_bytes_orig = base64.b64encode(buffer_orig.getvalue())
                            base64_string_orig = base64_bytes_orig.decode("utf-8")
                            image_content_orig.append({
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string_orig}",
                            })

                            # 下采样版本
                            v_ds = v
                            eval_logger.info(f"[Multi-Image-DOWN] 原始尺寸: {v_ds.size}")
                            w, h = v_ds.size
                            v_ds = v_ds.resize(
                                (w // scale_factor_default, h // scale_factor_default),
                                resample=Image.BICUBIC,
                            )
                            eval_logger.info(f"[Multi-Image-DOWN] 下采样后尺寸: {v_ds.size}")
                            base64_image_down = v_ds.convert("RGB")
                            buffer_down = BytesIO()
                            base64_image_down.save(buffer_down, format="JPEG")
                            base64_bytes_down = base64.b64encode(buffer_down.getvalue())
                            base64_string_down = base64_bytes_down.decode("utf-8")
                            image_content_down.append({
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string_down}",
                            })

                        msg_orig.append({
                            "role": "user",
                            "content": image_content_orig + [{"type": "text", "text": context}],
                        })
                        msg_down.append({
                            "role": "user",
                            "content": image_content_down + [{"type": "text", "text": context}],
                        })

                    # ---------- visual 类型不认识：都当纯文本 ----------
                    else:
                        msg_orig.append({"role": "user", "content": [{"type": "text", "text": context}]})
                        msg_down.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    # 完全没有 visuals，两套一样
                    msg_orig.append({"role": "user", "content": [{"type": "text", "text": context}]})
                    msg_down.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages_orig.append(msg_orig)
                messages_down.append(msg_down)

            # ======================
            # 这里就有两套 messages 了
            # ======================

            texts_orig = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_orig
            ]
            texts_down = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_down
            ]

            image_inputs_orig, video_inputs_orig = process_vision_info(messages_orig)
            image_inputs_down, video_inputs_down = process_vision_info(messages_down)
            # eval_logger.info(f"image_inputs:{image_inputs}")
            # image_inputs_downsample = self.resize_to_multiple(image_inputs, scale_factor=self.scale_factor)
            # --- Dual-path input construction ---
            # 1. Prepare High-res (for placeholders) and Low-res (for ViT computation) images
            # image_inputs_high = image_inputs_downsample # Original images as High (or upsampled if original is low)
            # image_inputs_low = None
            
            # if image_inputs is not None and len(image_inputs) > 0 and image_inputs[0] is not None:
                # Generate low-res images for ViT inference
                # image_inputs_low = self.lanczos_downsample(image_inputs, scale_factor=self.scale_factor)
            # else:
                # image_inputs_low = image_inputs

            # 2. Generate High Inputs (get input_ids, attention_mask with correct placeholder count)
            # These inputs contain the number of placeholders corresponding to the high-res image
            # inputs_high = self.processor(text=texts, images=image_inputs_high, videos=video_inputs, padding=True, return_tensors="pt")
            
            # 3. Generate Low Inputs (get pixel_values, image_grid_thw)
            # These inputs correspond to the small images actually sent to ViT
            # if image_inputs_low is not image_inputs_high:
                # inputs_low = self.processor(text=texts, images=image_inputs_low, videos=video_inputs, padding=True, return_tensors="pt")
            # else:
                # inputs_low = inputs_high

            # 4. Combine Inputs
            # Use High structure (input_ids) + Low visual data (pixel_values)
            # inputs_downsample = inputs_high

            # if image_inputs_low is not image_inputs_high:
                # inputs_downsample["pixel_values"] = inputs_low["pixel_values"]
                # inputs_downsample["image_grid_thw"] = inputs_low["image_grid_thw"]
                # eval_logger.info(f"pixel_values shape: {inputs_low['pixel_values'].shape}, image_grid_thw shape: {inputs_low['image_grid_thw'].shape}, image_grid_thw: {inputs_low['image_grid_thw']}")
                # Note: If videos were also downsampled, video_grid_thw would need replacement too
            
            # ==============================================================================
            # 将 inputs_downsample 存储到模型实例的属性中，避免通过参数传递
            # ==============================================================================
            # self.model.inputs_downsample = inputs_downsample
            # ==============================================================================

            # ==============================================================================
            # 创建 model_inputs，过滤掉 inputs_downsample（因为它现在通过属性传递）
            # ==============================================================================
            # model_inputs = {k: v for k, v in inputs.items() if k != "inputs_downsample"}
            # ==============================================================================
            video_inputs = video_inputs_orig
            if video_inputs is not None and len(video_inputs) > 0 and video_inputs[0] is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts_orig, images=image_inputs_orig, videos=video_inputs, padding=True, return_tensors="pt")
            inputs_downsample = self.processor(text=texts_down, images=image_inputs_down, videos=video_inputs, padding=True, return_tensors="pt")
            self.model.inputs_downsample = inputs_downsample

            image_inputs = image_inputs_orig
            
            # for key, value in inputs_downsample.items():
            #     new_key = f"{key}_downsample"
            #     inputs[new_key] = value

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id
            
            if self.config.Sparse_config is not None:
                # HACK
                if image_inputs is not None:
                    image_token_start_index = inputs.input_ids.tolist()[0].index(151655)
                if video_inputs is not None:
                    image_token_start_index = inputs.input_ids.tolist()[0].index(151656)
                image_token_end_index = inputs.input_ids.tolist()[0].index(151653)
                image_token_length = image_token_end_index - image_token_start_index
                self.config.Sparse_config['image_token_start_index'] = image_token_start_index
                self.config.Sparse_config['image_token_length'] = image_token_length
                # HACK

            # ---------------compute kv2-----------------------------
            if compute_flops_kv:
                meter.record_sample()
            # ---------------compute kv-----------------------------

            # ---------------compute time-------------
            torch.cuda.synchronize()  
            start_event.record()
            # ---------------compute time-------------
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            # ---------------test prefilling only-------------------
            # -------v2------
            # cont = self.model.generate(
            #     **inputs,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=pad_token_id,
            #     do_sample=True if gen_kwargs["temperature"] > 0 else False,
            #     temperature=gen_kwargs["temperature"],
            #     top_p=gen_kwargs["top_p"],
            #     num_beams=gen_kwargs["num_beams"],
            #     max_new_tokens=1,              #  generate 1 token
            #     use_cache=self.use_cache,
            # )
            # ---------------test prefilling only end-------------------

            
            # ---------------compute time-------------
            end_event.record()
            torch.cuda.synchronize()  # 等待 generate 完成
            total_infer_time += start_event.elapsed_time(end_event)  # 毫秒

            # sample_num+=1
            # if sample_num in [600,1200,2400]:
            #     total_seconds = total_infer_time / 1000
            #     minutes = int(total_seconds // 60)
            #     seconds = total_seconds % 60

            #     print(f"Total pure GPU inference time: {minutes} min {seconds:.2f} sec")
            # ---------------compute time-------------


            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #print(f"answers: {answers}")

            ## cuda memory free
            del inputs
            del cont
            torch.cuda.empty_cache()

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # ---------------compute kv3-----------------------------
        if compute_flops_kv:
            meter.stop()
            avg_flops, avg_kv_MB = meter.get_results()
            print(f"Average FLOPs: {avg_flops/1e9:.2f} GFLOPs, Average KV Cache: {avg_kv_MB:.2f} MB")
        # ---------------compute kv-----------------------------
        if self.rank == 0:  
            total_seconds = total_infer_time / 1000
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60

            print(f"Total pure GPU inference time: {minutes} min {seconds:.2f} sec")



        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")










