import os
import sys
import json
import subprocess
import numpy as np 
import pandas as pd
import folder_paths
import torch
from PIL import Image
import asyncio
import csv
from zhipuai_platform_video.reporting.console_workflow_callbacks import ConsoleWorkflowCallbacks
from zhipuai_platform_video.reporting.runner_callbacks import RunnerCallbacks
from zhipuai_platform_video.task import convert_image_to_video, convert_text_generator
from zhipuai_platform_video.result_task import video_pull_task



def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def init_dataset_return_datapath(prompt, images):
    if len(prompt) == 0:
         raise ValueError("Prompt cannot be empty")
 
    
    sample_data_dir = os.path.join(
        folder_paths.get_output_directory(), "zhipu/input/"
    )
    
    if not os.path.exists(sample_data_dir):
        os.makedirs(sample_data_dir)
    init_data = []
    if isinstance(images, torch.Tensor) and images.size(0) == 0:
            
        init_data.append({
            "input_text": prompt,
            "image_path": ""
        })
    else:
            
        frames = map(lambda x : Image.fromarray(tensor_to_bytes(x)), images)

        # Save each image and append to dataset
        for i, image in enumerate(frames):
            image_path = os.path.join(sample_data_dir, f"image_{i}.png")
            image.save(image_path)

            init_data.append({
                "input_text": prompt,
                "image_path": image_path
            })
        
    # Create a DataFrame with the initialized data
    init_dataset = pd.DataFrame(init_data)


    init_dataset.to_excel(f'{sample_data_dir}/dataset.xlsx', index=False)
    return f'{sample_data_dir}/dataset.xlsx'

class VideoReportGenerate:
    @classmethod
    def INPUT_TYPES(s): 
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }), 
                "prompt_num_threads": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "video_num_threads": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}), 
                
            }, 
            "optional": {
                "images": ("IMAGE",),

            }
        }

    @classmethod
    def IS_CHANGED(s, prompt, prompt_num_threads, video_num_threads, images=None,):
        return float("NaN") 
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt_report_path", "video_report_path",)
    FUNCTION = "video_report_generate"
    CATEGORY = "zhipuai/video"

    def video_report_generate(self, prompt, prompt_num_threads, video_num_threads, images=None,):
        input_excel = init_dataset_return_datapath(prompt, images)
        
        output_data_dir = os.path.join(
            folder_paths.get_output_directory(), "zhipu/output/"
        )
    
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        # Load the data
        level_contexts = pd.read_excel(input_excel)

        text_generator_strategy = {
            "input_text_key": "input_text",
            "num_threads": prompt_num_threads
        }
        callbacks = RunnerCallbacks(ConsoleWorkflowCallbacks())

        # Convert the image to video
        prompt_report: pd.DataFrame = asyncio.run(convert_text_generator(level_contexts=level_contexts,
                                                                        callbacks=callbacks,
                                                                        strategy=text_generator_strategy))
        # 合并level_contexts、prompt_report两个表格，重复字段会自动去重
        merged_report = pd.merge(level_contexts, prompt_report, on="input_text", how="left")
        # Save the video report
        prompt_report_path = os.path.join(output_data_dir, "prompt_report.csv")
        merged_report.to_csv(prompt_report_path, index=False)

        request_img = 'false'
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            request_img = 'false'
        else:
            request_img = 'true'

        image_to_video_strategy = {
            "image_path_key": "image_path",
            "video_prompt_key": "video_prompt",
            "num_threads": video_num_threads,
            "request_img": request_img == "true",
        }
        # Convert the image to video
        video_report: pd.DataFrame = asyncio.run(convert_image_to_video(level_contexts=merged_report,
                                                                        callbacks=callbacks,
                                                                        strategy=image_to_video_strategy))
        # Save the video report
        video_report_path = os.path.join(output_data_dir, "video_report.csv")
        video_report.to_csv(video_report_path, index=False)

        return (prompt_report_path, video_report_path)

class VideoReportPull:
    
    @classmethod
    def INPUT_TYPES(s): 

        return {
            "required": {
                "video_report_path": ("STRING", ),    
                
                "num_threads": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }
    @classmethod
    def IS_CHANGED(s, video_report_path, num_threads):
        return float("NaN") 

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_pull_report_path",)
    FUNCTION = "video_report_pull"
    CATEGORY = "zhipuai/video"

    def video_report_pull(self, video_report_path, num_threads):
        output_data_dir = os.path.join(
            folder_paths.get_output_directory(), "zhipu/output/"
        )
            
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        # Load the data
        level_contexts = pd.read_csv(video_report_path)

        video_strategy = {
            "video_task_id_key": "video_task_id",
            "num_threads": num_threads
        }
        callbacks = RunnerCallbacks(ConsoleWorkflowCallbacks())

        video_pull_report: pd.DataFrame = asyncio.run(video_pull_task(level_contexts=level_contexts,
                                                                    strategy=video_strategy,
                                                                    callbacks=callbacks))

        # Save the video report
        video_pull_report_path = os.path.join(output_data_dir, "video_pull_report.csv")
        video_pull_report.to_csv(video_pull_report_path, index=False)

        return (video_pull_report_path,)

 
 
class VideoReportData:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "file_path": ("STRING", {"forkInput": True}),  
            }
        }

    @classmethod
    def IS_CHANGED(s, file_path):
        return float("NaN") 
    
    RETURN_TYPES = ("DICT", "STRING", )
    RETURN_NAMES = ("data", "show_text", )

    FUNCTION = "csvinput"
    CATEGORY = "zhipuai/video"
    
    def csvinput(self, file_path):
      
        print(f"VideoReportData From File: Loading {file_path}")
        
        lists = []
        with open(file_path, "r") as csv_file:
            reader = csv.reader(csv_file)
    
            for row in reader:
                lists.append(row)
        
        return(lists,str(lists),)
    

NODE_CLASS_MAPPINGS = { 
    "VideoReportGenerate": VideoReportGenerate,
    "VideoReportPull": VideoReportPull, 
    "VideoReportData": VideoReportData, 
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoReportGenerate": "ZhipuAI VideoReportGenerate",
    "VideoReportPull": "ZhipuAI VideoReportPull", 
    "VideoReportData": "ZhipuAI VideoReportData", 
}