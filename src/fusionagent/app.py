
import torch
import os
import sys
import argparse
import time
import copy
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from peft import PeftModel
import gradio as gr
from gradio import ChatMessage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextStreamer, GenerationConfig
from transformers.utils import get_json_schema
import inspect
import yaml
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionagent.utils import XMLLikeTagParser, ConversationParser, SYSTEM_PROMPT_AGENT, SYSTEM_PROMPT_AGENT_FAST, TOOL_PROMPT
from fusionagent.WBModules import MODEL_MAPPING_DICT
from fusionagent.tool_func import RuntimeCtx, get_result_tool
from fusionagent.utils.file_utils import load_scoremats_from_h5
from fusionagent.configs.default_vid import get_vid_config
from fusionagent.configs.default_img import get_img_config
from fusionagent.data import VID_DATASET, build_interleave_dataset
from fusionagent.utils.load_center_utils import load_center_feat
from fusionagent.utils.eval_metrics import act_score_fusion_gradio


def parse_args():
    parser = argparse.ArgumentParser("FusionAgent Gradio Demo")
    parser.add_argument('--configs', type=str, default='./src/fusionagent/configs/train_config_test_ltcc.yaml')
    # parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the finetuned model.")
    parser.add_argument('--ckpt_path', type=str, default='./src/fusionagent/checkpoints/ccvid-grpo-200step-Rmetricv2_train-topkfusion_0831_1956', help="Path to the finetuned model.")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    configs = OmegaConf.to_container(configs, resolve=True)
    for k, v in configs.items():
        setattr(args, k, v)

    return args


class GradioAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load models and processor
        # check if trained with lora
        if os.path.exists(os.path.join(args.ckpt_path, "adapter_config.json")):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, 
                                                                    cache_dir=args.model_cache_dir,
                                                                    torch_dtype=torch.bfloat16,
                                                                    attn_implementation="flash_attention_2",
                                                                    device_map="cpu")
            lora_model = PeftModel.from_pretrained(self.model, args.ckpt_path)
            self.model = lora_model.merge_and_unload()
            print(f"Load lora ckpt from {args.ckpt_path}")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cpu",
            )
        self.model.to(self.device).eval()
        print("Model loaded successfully.")

        self.processor = AutoProcessor.from_pretrained(args.ckpt_path)
        
        # Load agent config
        if self.args.dataset in VID_DATASET:
            self.agent_config = get_vid_config(self.args)
        else:
            self.agent_config = get_img_config(self.args)

        # Initialize tools
        self.tools = [get_result_tool]
        self.tool_funcs = {func.__name__: func for func in self.tools}
        self._initialize_tool()
        print("Tools initialized successfully.")

        # Build system prompt
        tool_json_schema = [str(get_json_schema(tool)) for tool in self.tools]
        tool_prompt = TOOL_PROMPT.format(TOOL_SCHEMA="\n".join(tool_json_schema))
        model_type_dict = {k: v for k, v in MODEL_MAPPING_DICT.items() if k in RuntimeCtx.model_list}
        self.system_prompt = SYSTEM_PROMPT_AGENT.format(TOOL_PROMPT=tool_prompt, MODEL_TYPE_DICT=model_type_dict)
        self.system_prompt_fast = SYSTEM_PROMPT_AGENT_FAST.format(TOOL_PROMPT=tool_prompt, MODEL_TYPE_DICT=model_type_dict)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            # temperature=0,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

    @staticmethod
    def load_and_resize_image(img_data, resize_h=256, resize_w=128):
        """
        Load and resize image data. If img_data is a PIL image, it will be resized directly.
        Otherwise, it will be converted to a PIL image and then resized.
        """
        if isinstance(img_data, Image.Image):
            img = img_data
        else:
            img = Image.fromarray(np.array(img_data))
        return img.resize((resize_w, resize_h))

    def _initialize_tool(self):
        from fusionagent.WBModules import Model_Wrapper  # lazy import
        
        # from fusionagent.data import build_raw_dataset
        _, self.query_dataset, _, self.raw_dataset = build_interleave_dataset(self.agent_config)
        print("Dataset loaded successfully.")

        BACKBONE_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'WBModules', f'model_cfg_{self.agent_config.DATA.DATASET}.yaml'), 'r'))
        model_names: list[str] = BACKBONE_CFG["model_list"]
        tool_model_list: dict[str, torch.nn.Module] = {}
        dataset_type = "vid" if getattr(self.agent_config.DATA, "DATASET", None) in VID_DATASET else "img"

        for mode in model_names:
            model = Model_Wrapper(
                mode=mode,
                backbone_cfg=BACKBONE_CFG,
                config=self.agent_config,
                is_training=False,
                dataset_type=dataset_type,
            )
            model.eval().to(self.device)
            tool_model_list[mode] = model
            
        # Center features for retrieval-based tools
        # NOTE: This is the center features for the training set
        center_feats = load_center_feat(model_names, self.raw_dataset, 'cpu')
            
        test_feats = load_scoremats_from_h5(f'./src/fusionagent/test_feats/scoremats_{self.agent_config.DATA.DATASET}.h5', model_names)
        test_feats['dataset_name'] = self.agent_config.DATA.DATASET
        
        RuntimeCtx.set(
            model_list=tool_model_list, 
            center_feats=center_feats, 
            dataset_name=self.agent_config.DATA.DATASET, 
            test_feats=test_feats,
        )
        
    def execute_tool_from_response(self, response: list[dict], sequence: dict) -> any:
        last_result = None
        for call in response:
            tool_name = call.get("name")
            tool_kwargs = call.get("tool_kwargs", {})
            
            RuntimeCtx.set(sequence=sequence)
            
            tool_func = self.tool_funcs.get(tool_name)
            if tool_func is None:
                last_result = {"error": f"Unknown tool: {tool_name}"}
                continue
            try:
                sig = inspect.signature(tool_func)
                input_args = {arg: tool_kwargs.get(arg) for arg in sig.parameters}
                last_result = tool_func(**input_args)
            except Exception as e:
                last_result = {"error": str(e), "tool": tool_name}
        return last_result

    def run_from_paths(self, image_paths, text_query, history, prompt_mode='cot'):
        pil_images = [self.load_and_resize_image(Image.open(p)) for p in image_paths]
        body_image_keys = [os.path.basename(p) for p in image_paths]
        # dummy values for other things
        face_pil_images = []
        face_image_keys = []
        subject_id = "unknown"
        has_face = [False] * len(pil_images)

        # call _run
        yield from self._run(pil_images, face_pil_images, body_image_keys, face_image_keys, subject_id, has_face, text_query, history, index=None, prompt_mode=prompt_mode)

    def run_from_index(self, index, text_query, history, prompt_mode='cot'):
        data_item = self.query_dataset[index]

        # as in evaluation_loop
        num_samples = self.agent_config.num_sample
        body_image_keys = data_item['body_image_keys'][:num_samples]
        face_image_keys = data_item['face_image_keys'][:num_samples]
        has_face = data_item['has_face'][:num_samples]
        subject_id = data_item['subject_id']

        body_pil_images = [self.load_and_resize_image(self.raw_dataset.body_data[k]) for k in body_image_keys]
        face_pil_images = [self.load_and_resize_image(self.raw_dataset.face_data[k]) for k in face_image_keys]
        
        pil_images = body_pil_images
        
        yield from self._run(pil_images, face_pil_images, body_image_keys, face_image_keys, subject_id, has_face, text_query, history, index, prompt_mode)

    def _run(self, pil_images, face_pil_images, body_image_keys, face_image_keys, subject_id, has_face, text_query, history, index, prompt_mode='cot'):
        history = []

        # Prepare user message
        user_content = [{"type": "video", "video": pil_images}]
        user_content.append({"type": "text", "text": text_query})
        
        if prompt_mode == 'fast':
            system_prompt = self.system_prompt_fast
        else:
            system_prompt = self.system_prompt
            
        current_messages = [
            {"role": "system", "content": system_prompt},
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]

        history.append(ChatMessage(role="user", content=text_query))
        yield history

        with torch.inference_mode():
            model_list = []
            for _ in range(self.agent_config.max_turns):
                prompt_text = self.processor.apply_chat_template(
                    current_messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.processor(
                    text=[prompt_text],
                    videos=[pil_images], # The processor expects a list of lists of images for videos
                    return_tensors="pt",
                ).to(self.device)

                gen_out = self.model.generate(**inputs, generation_config=self.generation_config)
                response_ids = gen_out[0, inputs['input_ids'].shape[1]:]
                response_text = self.processor.decode(response_ids, skip_special_tokens=True)

                current_messages.append({"role": "assistant", "content": response_text})

                # Parse and execute tools if any
                xml_parser = XMLLikeTagParser(response_text)                
                # history.append(ChatMessage(role="assistant", content=display_text))
                # yield history
                    
                if 'tool_call' in xml_parser.get_ordered_tags():
                    # Stream thinking in the response
                    if 'think' in xml_parser.get_ordered_tags():
                        think_content = xml_parser.get_contents('think')
                        if think_content:
                            display_text = str("<think>" + think_content[0] + "</think>").replace("<", "&lt;").replace(">", "&gt;")
                            for updated_history in self.stream_response(history, display_text):
                                yield updated_history
                        
                    tool_input = xml_parser.parse_tool_calls()
                    
                    # This part needs to be adapted from your dataset logic
                    # We create a dummy sequence for tool execution
                    sequence = {
                         "body_pil_image": pil_images,
                         "face_pil_image": face_pil_images,
                         "body_image_keys": body_image_keys,
                         "face_image_keys": face_image_keys,
                         "subject_id": subject_id,
                         "has_face": has_face,
                    }
                    
                    tool_result = self.execute_tool_from_response(tool_input, sequence)
                    # Provide the true result to the agent as we using training gallery
                    # Get the model name from the tool result and return the stored score matrix
                    if tool_result is not None and isinstance(tool_result, dict):
                        if 'model_name' in tool_result:
                            model_name = tool_result['model_name']
                            score_mat = RuntimeCtx.test_feats['score_mats'][model_name][index]
                            RuntimeCtx.put(f'{model_name}_similarity_matrix', score_mat.unsqueeze(0))
                            pred_identity_id = str(RuntimeCtx.test_feats['g_pids'][score_mat.argmax()].item())
                            tool_result['pred_identity_id'] = f'{RuntimeCtx.dataset_name}_{pred_identity_id}'
                    tool_result_str = str(tool_result)
                    
                    tool_message = {"role": "system", "content": f"<tool_result>{tool_result_str}</tool_result>"}
                    current_messages.append(tool_message)

                    # Update chatbot with tool result
                    history.append( {"role": "assistant", "content": "", "metadata": {"title": f"🛠️ Tool Output: `{model_name}`"}})
                    yield history
                    time.sleep(1)
                    history[-1]['content'] = f"```json\n{tool_result_str}\n```"
                    yield history
                    model_list.append(model_name)                    
                    continue
                elif 'answer' in xml_parser.get_ordered_tags():
                    # TODO: use fused score matrix to get the final answer
                    similarity_matrices = {}
                    for model_name in model_list:
                        similarity_matrices[model_name] = RuntimeCtx.get(f'{model_name}_similarity_matrix').cpu().numpy()
                    fused_score_mat = act_score_fusion_gradio(similarity_matrices, [model_list[0]], top_k=self.args.top_k, norm_method='zscore')
                    pred_identity_id = str(RuntimeCtx.test_feats['g_pids'][fused_score_mat.argmax()].item())
                    # Replace the answer in the response text
                    response_text = response_text.replace(xml_parser.get_contents('answer')[0], f"{RuntimeCtx.dataset_name}_{pred_identity_id}")
                            
                    # Stream whole response
                    display_text = str(response_text).replace("<", "&lt;").replace(">", "&gt;")
                    for updated_history in self.stream_response(history, display_text):
                        yield updated_history
                    break 
                else: # No tool call, no answer, maybe an error or intermediate thought
                    # Let's just break for simplicity in the demo
                    display_text = str(response_text).replace("<", "&lt;").replace(">", "&gt;")
                    for updated_history in self.stream_response(history, display_text):
                        yield updated_history
                    break
            RuntimeCtx.clear_storage()
        yield history
    
    def stream_response(self, history: list[ChatMessage], message: str):
        history.append({"role": "assistant", "content": ""})
        for character  in message:
            history[-1]['content'] += character 
            time.sleep(0.01)
            yield history
        return history

def main():
    args = parse_args()
    agent = GradioAgent(args)

    # CSS for styling chatbot messages


    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FusionAgent Demo")
        # gr.Markdown("Select a tab to either search by index from the pre-loaded dataset or by uploading your own images.")

        with gr.Tabs():
            with gr.TabItem("1:N Search (Index)"):
                gr.Markdown("This tab is for querying from the whole-body biometric dataset by index.")
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(f"### Query from {agent.agent_config.DATA.DATASET} Dataset")
                        index_slider = gr.Slider(
                            minimum=0, 
                            maximum=len(agent.query_dataset) - 1 if agent.query_dataset else 0, 
                            value=0, 
                            step=1, 
                            label="Query Index"
                        )
                        gt_label_index = gr.Textbox(label="Ground Truth Subject ID", interactive=False)
                        text_input_index = gr.Textbox(label="User Input", placeholder="Identify the person in the video from the dataset.", value="Identify the person in the video from the dataset.")
                        cot_mode_index = gr.Checkbox(label="CoT Mode", value=True)
                        with gr.Row():
                            submit_btn_index = gr.Button("Submit", variant="primary")
                            clear_btn_index = gr.Button("Clear History")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Visualized Face Images")
                        face_gallery = gr.Gallery(height=112, columns=agent.agent_config.num_sample, object_fit="contain")
                        gr.Markdown("#### Visualized Body Images")
                        body_gallery = gr.Gallery(height=256, columns=agent.agent_config.num_sample, object_fit="contain")
                
                chatbot_index = gr.Chatbot(label="Conversation", height=800, bubble_full_width=False, type="messages", show_copy_button=True)
                
                def on_index_change(index):
                    index = int(index)
                    data_item = agent.query_dataset[index]
                    subject_id = data_item['subject_id']
                    num_samples = agent.agent_config.num_sample
                    body_image_keys = data_item['body_image_keys'][:num_samples]
                    face_image_keys = data_item['face_image_keys'][:num_samples]

                    body_pil_images = [Image.fromarray(np.array(agent.raw_dataset.body_data[k])) for k in body_image_keys]
                    face_pil_images = [Image.fromarray(np.array(agent.raw_dataset.face_data[k])) for k in face_image_keys] if face_image_keys else []
                    
                    return body_pil_images, face_pil_images, subject_id

                def on_submit_index(index, text, history, cot_mode):
                    if not text:
                        gr.Warning("Please enter a question.")
                    else:
                        prompt_mode = "cot" if cot_mode else "fast"
                        for chatbot_update in agent.run_from_index(int(index), text, history, prompt_mode):
                            yield chatbot_update
                
                index_slider.change(on_index_change, inputs=[index_slider], outputs=[body_gallery, face_gallery, gt_label_index])
                
                submit_btn_index.click(
                    on_submit_index,
                    inputs=[index_slider, text_input_index, chatbot_index, cot_mode_index],
                    outputs=[chatbot_index]
                )

                def clear_history_index():
                    return []

                clear_btn_index.click(clear_history_index, outputs=[chatbot_index])
                
                # Initial load
                demo.load(on_index_change, inputs=[index_slider], outputs=[body_gallery, face_gallery, gt_label_index])

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)

if __name__ == "__main__":
    main()