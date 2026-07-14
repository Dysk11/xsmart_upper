# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np
import argparse
from .utils import operators
from .utils.db_postprocess import DBPostProcess, DetPostProcess

DET_INPUT_SHAPE = [480, 480] # h,w

RKNN_PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': {
                    'image_shape': DET_INPUT_SHAPE
                }
         }, 
        {
            'NormalizeImage': 
            {
                    'std': [1., 1., 1.],
                    'mean': [0., 0., 0.],
                    'scale': '1.',
                    'order': 'hwc'
            }
        }
        ]

POSTPROCESS_CONFIG = {
    'DBPostProcess':{
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
    }
}

class TextDetector:
    def __init__(self, args) -> None:
        self.model, self.framework = setup_model(args)
        self.preprocess_funct = []
        for item in RKNN_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)

        # RKNNLite 2.3.x requires the static NHWC model's batch dimension.
        if img['image'].ndim == 3:
            img['image'] = np.expand_dims(img['image'], axis=0)

        return img

    def run(self, img):
        model_input = self.preprocess({'image':img})
        output = self.model.run([model_input['image']])

        preds = {'maps' : output[0].astype(np.float32)}
        result = self.db_postprocess(preds, model_input['shape'])

        output = self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)
        return output

def setup_model(args):
    model_path = args.det_model_path
    if not model_path.endswith('.rknn'):
        raise ValueError("PPOCR detector requires an RKNN model: {}".format(model_path))
    platform = 'rknn'
    from ..py_utils.rknn_executor import RKNN_model_container
    model = RKNN_model_container(model_path, args.target, args.device_id, args.core_mask)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform
