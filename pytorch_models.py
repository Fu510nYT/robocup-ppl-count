#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch

class TorchModel(object):
    def __init__(self, torch_home: str) -> None:
        if torch_home is None:
            if "TORCHHUB_DIR" in os.environ:
                torch_home = os.environ["TORCHHUB_DIR"]
            else:
                torch_home = "~/models/pytorch/hub"
                
        os.environ["TORCH_HOME"] = torch_home
        self.device = "cpu"
    
    def BGR2Tensor(self, frame):
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        img = np.array(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        img = img.to(self.device)
        return img

class Yolov5(TorchModel):
    def __init__(self, torch_home: str = None) -> None:
        super().__init__(torch_home)
        self.net = torch.hub.load("ultralytics/yolov5", "yolov5s", device="cpu", force_reload=True)
        self.net.eval().to(self.device)
        self.labels = self.net.names

    def forward(self, frame):
        img = frame.copy()
        out = self.net(img)

        res = []
        for x1, y1, x2, y2, pred, index in out.xyxy[0]:
            if pred < 0.7: continue
            x1, y1, x2, y2, index = map(int, (x1, y1, x2, y2, index))
            res.append([0, index, pred, x1, y1, x2, y2])
        return res
