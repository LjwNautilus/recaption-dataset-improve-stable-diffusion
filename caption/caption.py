import argparse
import math
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
from rich.progress import (Progress, TextColumn, BarColumn,
                           MofNCompleteColumn, TaskProgressColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
# import warnings

import jsonlines
from datasets import load_dataset

from sharegpt4v import ShareGPT4V

# warnings.filterwarnings("error", category=UserWarning)

def clean_caption(caption: str):
    caption = caption.replace('\n', '')
    caption = caption.replace('*', '')
    return caption

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, required=True)
    parser.add_argument('data_dir', type=str, required=True)
    parser.add_argument('json_file', type=str, required=True)
    parser.add_argument('batch_size', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # model_path = '/home/luojingwu/models/ShareGPT4V-7B'
    model_path = args.model_path
    device = 'cuda'
    # data_dir = '/data1/luojingwu/dataset/COCO_captions_validation'
    data_dir = args.data_dir

    dataset = load_dataset(data_dir, split='validation')

    prompt = '''You are a powerful image captioner. Create detailed captions describing the contents of the given image. Include the object types and colors, counting the objects, object actions, precise object locations, texts, doublechecking relative positions between objects, etc. Instead of describing the imaginary content, only describing the content one can determine confidently from the image.'''

    model = ShareGPT4V(model_path, device=device)

    batch_size = args.batch_size

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        '[', TimeElapsedColumn(),
        '<', TimeRemainingColumn(), ']',
    )

    target_jsonl_name = 'coco_caption.jsonl'
    total = 5000
    bias = 0

    with (jsonlines.open(f'{data_dir}/{target_jsonl_name}', 'a') as writer,
          progress):
        writer._flush = True

        task = progress.add_task("[b cyan]Captioning...",
                                 total=total)
        for i in range(bias, bias + math.ceil(total / batch_size)):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size

            batch_data = dataset[begin_idx:end_idx]
            batch_img = [batch_data['image'][i].convert('RGB')
                         for i in range(len(batch_data['image']))]
            batch_raw_captions = batch_data['sentences_raw']
            batch_raw_captions = tuple(batch_raw_captions[i][0]
                                       for i in range(len(batch_raw_captions)))
            
            caption_list = model.generate(batch_img, prompt,
                                          stream_out=False)
            for i in range(len(caption_list)):
                share_caption = clean_caption(caption_list[i])
                writer.write({'raw_caption': batch_raw_captions[i],
                              'share_caption': share_caption})
            progress.update(task, advance=len(caption_list))