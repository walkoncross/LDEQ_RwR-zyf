import argparse
import json
import os.path as osp
import time
from typing import Optional, Union

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis

from datasets.WFLW_V.helpers import *
from models.ldeq import LDEQ, weights_init
from utils.helpers import *
from utils.loss_function import *
from utils.normalize import HeatmapsToKeypoints

heatmaps_to_keypoints = HeatmapsToKeypoints()


class DEQInference(object):
    def __init__(self, args):
        self.args = args

        ## Model
        ckpt = torch.load(args.landmark_model_weights, map_location="cpu")
        self.train_args = ckpt["args"]
        # self.train_args.solver = "fpi"
        # self.train_args.max_iters = 7
        self.train_args.stochastic_max_iters = (
            False  # use maximum iters at inference time so perf repeatable
        )
        print(f"--> Train args: \n{json.dumps(vars(self.train_args), indent=2)}")
        # with open("./checkpoints/WFLW_ckpt_args.json", "w") as fp:
        #     json.dump(vars(self.train_args), fp, indent=2)

        self.model = LDEQ(self.train_args).to(self.args.device)
        self.model.apply(weights_init)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        self.model.to(self.args.device)
        print(
            f"--> Restored weights for {self.train_args.landmark_model_name} from {self.args.landmark_model_weights}"
        )

        ## Video stuff
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def get_z0(self, batch_size):
        if self.train_args.z0_mode == "zeros":
            return torch.zeros(
                batch_size,
                self.train_args.z_width,
                self.train_args.heatmap_size,
                self.train_args.heatmap_size,
                device=self.args.device,
            )
        else:
            raise NotImplementedError

    def predict(
        self,
        img,
        bboxes: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    ):
        """test code adapted from https://github.com/starhiking/HeatmapInHeatmap"""
        self.model.eval()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if bboxes is None:
            img = cv2.resize(img, (256, 256))
            print(f"--> img.shape: {img.shape}")
            img = self.normalize(img)
            img = img.unsqueeze(0).to(args.device)

            with torch.no_grad():
                output = self.model(
                    img,
                    mode=self.train_args.model_mode,
                    args=self.train_args,
                    z0=self.get_z0(img.shape[0]),
                )
            # print(f"--> output: {output}")
            pred_keypoints = output["keypoints"].cpu().numpy() * 256
        else:
            pred_keypoints = []
            for bbox in bboxes:
                print(f"--> bbox: {bbox}")
                transform_matrix = get_transform_from_bbox(
                    bbox, extra_scale=1.2, target_im_size=256
                )
                face_np = cv2.warpAffine(
                    img,
                    transform_matrix,
                    (256, 256),
                    flags=cv2.INTER_LINEAR,
                )

                face_torch = self.normalize(face_np).unsqueeze(0).to(self.args.device)

                with torch.no_grad():
                    output = self.model(
                        face_torch,
                        mode=self.train_args.model_mode,
                        args=self.train_args,
                        z0=self.get_z0(face_torch.shape[0]),
                    )

                # print(f"--> output: {output}")

                kpt_preds = apply_affine_transform_to_kpts(
                    output["keypoints"].cpu().numpy().squeeze() * 256,
                    transform_matrix,
                    inverse=True,
                )

                pred_keypoints.append(kpt_preds)

            pred_keypoints = np.stack(pred_keypoints)

        # pred_keypoints = pred_keypoints * 256
        return pred_keypoints


def main(args):
    if args.device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    t0 = time.time()
    solver = DEQInference(args)
    print(f"Model loaded in {format_time(time.time() - t0)}")

    img_ori = cv2.imread(args.input)
    bboxes = []

    if args.detector == "dlib":
        detector = dlib.get_frontal_face_detector()

        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        rects = detector(gray)
        print(f"--> rects: {rects}")

        for rect in rects:
            bboxes.append(
                [
                    rect.left(),
                    rect.top(),
                    rect.right(),
                    rect.bottom(),
                ]
            )
    elif args.detector == "insightface":
        app = FaceAnalysis(
            providers=[
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(img_ori)

        for face in faces:
            bboxes.append(face.bbox)
    else:
        raise ValueError(f"Unsupported detector type: {args.detector}")

    if bboxes == []:
        bboxes = None
    else:
        bboxes = np.array(bboxes)

    print(f"--> bboxes: {bboxes}")

    t0 = time.time()
    landmarks = solver.predict(img_ori, bboxes=bboxes)

    total_process_time = time.time() - t0
    num_faces = len(bboxes) if bboxes is not None else 1

    print(f"--> landmakrs.shape: {landmarks.shape}")
    # print(landmarks)

    print(f"--> Total process time: {format_time(total_process_time)}")
    print(f"--> Process time per face: {format_time(total_process_time/num_faces)}")

    if args.device == "cuda":
        print(
            f"Max mem: {torch.cuda.max_memory_allocated(device='cuda') / (1024**3):.1f} GB"
        )
    elif args.device == "mps":
        print(f"Max mem: {torch.mps.current_allocated_memory() / (1024**3):.1f} GB")
    else:
        print("Max mem: N/A")

    if bboxes is not None:
        for bbox in bboxes:
            res_img = draw_bbox(bbox, img_ori, thickness=2)
    res_img = draw_landmark(landmarks.reshape(-1, 2), img_ori, linewidth=1)
    cv2.imshow("result", res_img)
    cv2.waitKey(0)

    save_path = osp.splitext(args.input)[0] + "-ldeq.jpg"
    cv2.imwrite(save_path, res_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEQ Inference")
    parser.add_argument(
        "--landmark_model_weights",
        type=str,
        default="./checkpoints/WFLW-final.pth.tar",
        help="path to landmark model weights",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../WFLW/test_expression/wflw_test_expression_1.jpg",
        help="path to WFLW dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="cpu, mps or cuda",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="insightface",
        help="face detector, choose from 'dlib', 'insightface'",
    )

    args = parser.parse_args()
    print(f"--> args: \n{args}")

    print("\nStarting...")
    main(args)
