import os
from typing import Dict, Union, Callable

import numpy as np
import torch
import clip
import mmcv
import PIL
from PIL import Image
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input)
import traceback


class ExtractCLIP(torch.nn.Module):

    def __init__(self, args, external_call=False):
        super(ExtractCLIP, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.extraction_fps = args.extraction_fps
        self.on_extraction = args.on_extraction
        self.external_call = external_call
        if external_call is False:
            self.output_direct = args.output_direct
            if self.output_direct is True:
                self.output_path = args.output_path
            else:
                self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        """
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        """
        device = indices.device

        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
        if self.feature_type == 'CLIP-ViT-B/32':
            model, preprocess = clip.load("ViT-B/32", device=device)
        elif self.feature_type == 'CLIP-ViT-B/16':
            model, preprocess = clip.load("ViT-B/16", device=device)
        elif self.feature_type == 'CLIP-RN50x16':
            model, preprocess = clip.load("RN50x16", device=device)
        elif self.feature_type == 'CLIP-RN50x4':
            model, preprocess = clip.load("RN50x4", device=device)
        elif self.feature_type == 'CLIP-RN101':
            model, preprocess = clip.load("RN101", device=device)
        elif self.feature_type == 'CLIP-RN50':
            model, preprocess = clip.load("RN50", device=device)
        else:
            raise NotImplementedError
        model.eval()

        feats_list = []
        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, model, preprocess, self.path_list[idx])
                if self.external_call is False:
                    action_on_extraction(feats_dict, self.path_list[idx], self.output_path,
                                         self.on_extraction, self.output_direct)
                else:
                    feats_list.append(feats_dict)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]} with error (↑). Continuing extraction')
                traceback.print_exc()

            # update tqdm progress bar
            self.progress.update()
        return feats_list

    def extract(self, device: torch.device, model: torch.nn.Module,
                preprocess_func: Callable[[PIL.Image], torch.Tensor],
                video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.
           Note that this function won't generate any tmp files

        Arguments:
            device {torch.device}
            model {torch.nn.Module}

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: 'features_nme', 'fps', 'timestamps_ms'
        """

        def _process_frame(frame):
            frame = Image.fromarray(video.get_frame(frame))
            frame = preprocess_func(frame)  # C H W
            return frame

        video = mmcv.VideoReader(str(video_path))  # H W C
        fps, frame_cnt = video.fps, video.frame_cnt
        mspf = 0.001 / fps  # ms per frame

        assert self.extraction_fps is not None
        samples_num = int(frame_cnt / fps * self.extraction_fps)  # get num of sample frame to be extracted
        samples_ix = np.linspace(0, video.frame_cnt - 1, samples_num).astype(int)
        timestamps_ms = [i * mspf for i in samples_ix]

        frames = torch.stack(list(map(lambda x: _process_frame(x), samples_ix))).to(device)  # T C H W
        with torch.no_grad():
            features = model.encode_image(frames)  # T E

        features_with_meta = {
            self.feature_type: features.cpu().numpy(),
            'fps': np.array(fps),
            'timestamps_ms': np.array(timestamps_ms)
        }
        return features_with_meta
