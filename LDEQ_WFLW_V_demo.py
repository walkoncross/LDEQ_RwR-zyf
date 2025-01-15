import argparse
import concurrent.futures
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from datasets.WFLW_V.helpers import *
from models.ldeq import LDEQ, weights_init
from utils.helpers import *
from utils.loss_function import video_NME_NMJ
from utils.normalize import HeatmapsToKeypoints

heatmaps_to_keypoints = HeatmapsToKeypoints()


class DEQInference(object):
    def __init__(self, args):
        self.args = args

        ## Model
        ckpt = torch.load(args.landmark_model_weights, map_location="cpu")
        self.train_args = ckpt["args"]
        self.train_args.stochastic_max_iters = (
            False  # use maximum iters at inference time so perf repeatable
        )
        # print(f"ckpt args: {self.train_args}")
        print(f"Train args: {vars(self.train_args)}")

        # with open("./checkpoints/WFLWV_ckpt_args.json", "w") as fp:
        #     json.dump(vars(self.train_args), fp, indent=2)

        self.model = LDEQ(self.train_args).to(self.args.device)
        self.model.apply(weights_init)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        print(
            f"Restored weights for {self.train_args.landmark_model_name} from {self.args.landmark_model_weights}"
        )

        ## Video stuff
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

    ###################
    # Sequential evaluation = one video at a time frame by frame.
    # This is convenient to plot and debug but slow.
    # Use batching (further down) to go faster

    def test_single_videos_sequential(
        self,
        video_path,
        RWR,
        plot=False,
    ):
        self.train_args.verbose_solver = self.args.verbose_solver
        if RWR:
            self.train_args.max_iters = self.args.rwr_max_iters
            self.train_args.take_one_less_inference_step = (
                self.args.rwr_take_one_less_inference_step
            )

        t0 = time.time()
        assert os.path.isfile(video_path)
        video_capture = cv2.VideoCapture(video_path)
        n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = [video_capture.read()[1] for _ in range(n_frames)]
        video_capture.release()
        oracle_kpts = np.load(
            os.path.join(self.args.video_oracle_kpts_folder_path, f"{video_ID}.npy")
        )
        bboxes = np.load(
            os.path.join(self.args.video_bboxes_folder_path, f"{video_ID}.npy")
        )

        if RWR:
            NME, NMJ = self._test_single_video_sequential_RWR(
                video_frames,
                oracle_kpts,
                bboxes,
                plot=plot,
            )
            print(
                f"video {video_ID} (NME, NMJ) = ({NME:02.2f}, {NMJ:02.2f}) --- process time: {format_time(time.time() - t0)}"
            )

        else:
            NME, NMJ = self._test_single_video_sequential(
                video_frames,
                oracle_kpts,
                bboxes,
                plot=plot,
            )
            print(
                f"video {video_ID} (NME, NMJ) = ({NME:02.2f}, {NMJ:02.2f}) --- process time: {format_time(time.time() - t0)}"
            )

        NMEs.append(NME)
        NMJs.append(NMJ)

        print("\n\n\n----------------------")
        print(f"RWR = {RWR} for {video_idx + 1} videos:")
        print(f"avg (NME, NMJ) = {np.mean(NMEs):02.2f}, {np.mean(NMJs):02.2f}")

    def test_all_videos_sequential(
        self,
        RWR,
        plot=False,
        num_video=4,
        start_video_idx=0,
    ):
        with open(self.args.video_list_file_path, "r") as f:
            video_IDs = [n.strip() for n in f.readlines()]

        (
            NMEs,
            NMJs,
            abs_diffs,
            rel_diffs,
            fp_losses,
            reg_losses,
            NMEs_no_reg,
            NMJs_no_reg,
        ) = [], [], [], [], [], [], [], []

        self.train_args.verbose_solver = self.args.verbose_solver
        if RWR:
            self.train_args.max_iters = self.args.rwr_max_iters
            self.train_args.take_one_less_inference_step = (
                self.args.rwr_take_one_less_inference_step
            )

        for video_idx, video_ID in enumerate(
            video_IDs[start_video_idx : start_video_idx + num_video]
        ):
            t0 = time.time()
            path_to_video = os.path.join(
                self.args.video_frames_folder_path, f"{video_ID}.mp4"
            )
            assert os.path.isfile(path_to_video)
            video_capture = cv2.VideoCapture(path_to_video)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames = [video_capture.read()[1] for _ in range(n_frames)]
            video_capture.release()
            oracle_kpts = np.load(
                os.path.join(self.args.video_oracle_kpts_folder_path, f"{video_ID}.npy")
            )
            bboxes = np.load(
                os.path.join(self.args.video_bboxes_folder_path, f"{video_ID}.npy")
            )

            if RWR:
                NME, NMJ = self._test_single_video_sequential_RWR(
                    video_frames,
                    oracle_kpts,
                    bboxes,
                    plot=plot,
                )
                print(
                    f"video {video_ID} (NME, NMJ) = ({NME:02.2f}, {NMJ:02.2f}) --- process time: {format_time(time.time() - t0)}"
                )

            else:
                NME, NMJ = self._test_single_video_sequential(
                    video_frames,
                    oracle_kpts,
                    bboxes,
                    plot=plot,
                )
                print(
                    f"video {video_ID} (NME, NMJ) = ({NME:02.2f}, {NMJ:02.2f}) --- process time: {format_time(time.time() - t0)}"
                )

            NMEs.append(NME)
            NMJs.append(NMJ)

        print("\n\n\n----------------------")
        print(f"RWR = {RWR} for {video_idx + 1} videos:")
        print(f"avg (NME, NMJ) = {np.mean(NMEs):02.2f}, {np.mean(NMJs):02.2f}")

    @torch.no_grad()
    def _test_single_video_sequential(self, video_frames, oracle_kpts, bboxes, plot):
        """This is our baseline. We process each frame individually for both fine and coarse bboxes"""

        n_frames = len(video_frames)
        pred_frame_kpts = np.zeros((n_frames, 98, 2))
        self.prev_NMJs = None  # only to debug NMJ metric

        for frame_idx in range(n_frames):
            z0 = self.get_z0(1)

            transform_matrix = get_transform_from_bbox(
                bboxes[frame_idx], extra_scale=1.2, target_im_size=256
            )
            face_np = cv2.warpAffine(
                video_frames[frame_idx],
                transform_matrix,
                (256, 256),
                flags=cv2.INTER_LINEAR,
            )
            face_torch = (
                self.normalize(cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                .unsqueeze(0)
                .to(self.args.device)
            )
            output = self.model(
                face_torch, mode=self.train_args.model_mode, args=self.train_args, z0=z0
            )
            kpt_preds = apply_affine_transform_to_kpts(
                output["keypoints"].cpu().numpy().squeeze() * 256,
                transform_matrix,
                inverse=True,
            )
            pred_frame_kpts[frame_idx] = kpt_preds

            ##plot
            if plot:
                frame = video_frames[frame_idx]
                frame = draw_bbox(bboxes[frame_idx], frame, thickness=2)
                frame = draw_landmark(
                    oracle_kpts[frame_idx], frame, bgr=(0, 255, 0), linewidth=1
                )
                frame = draw_landmark(kpt_preds, frame, bgr=(0, 0, 255), linewidth=1)
                cv2.imshow("predictions vs. gnd truth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break  # press q key to break

        NME, NMJ = video_NME_NMJ(pred_frame_kpts, oracle_kpts)

        return NME, NMJ

    @torch.no_grad()
    def _test_single_video_sequential_RWR(
        self, video_frames, oracle_kpts, bboxes, plot
    ):
        """
        Use z_star from prev frame as z0 of current frame
        """
        n_frames = len(video_frames)
        pred_frame_kpts = np.zeros((n_frames, 98, 2))
        prev_z_star = None

        for frame_idx in range(n_frames):
            transform_matrix = get_transform_from_bbox(
                bboxes[frame_idx], extra_scale=1.2, target_im_size=256
            )
            face_np = cv2.warpAffine(
                video_frames[frame_idx],
                transform_matrix,
                (256, 256),
                flags=cv2.INTER_LINEAR,
            )
            face_torch = (
                self.normalize(cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                .unsqueeze(0)
                .to(self.args.device)
            )

            output = self.model(
                face_torch,
                mode=self.train_args.model_mode,
                args=self.train_args,
                z0=prev_z_star if prev_z_star is not None else self.get_z0(1),
            )
            preds = apply_affine_transform_to_kpts(
                output["keypoints"].cpu().numpy().squeeze() * 256,
                transform_matrix,
                inverse=True,
            )
            pred_frame_kpts[frame_idx] = preds

            prev_z_star = output["z_star"]

            ##------------------- plot
            if plot:
                frame = video_frames[frame_idx]
                frame = draw_bbox(bboxes[frame_idx], frame, thickness=2)
                frame = draw_landmark(
                    oracle_kpts[frame_idx], frame, bgr=(0, 255, 0), linewidth=1
                )
                frame = draw_landmark(preds, frame, bgr=(0, 0, 255), linewidth=1)
                cv2.imshow("Oracle v.s. fine predictions", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break  # press q key to break

        NME, NMJ = video_NME_NMJ(pred_frame_kpts, oracle_kpts)

        return NME, NMJ

    ###################
    # LDEQ evaluation on video can be quite slow
    # Batching isn't trivial because in rwr we want to use previous frames to infer new frames.
    # Therefore we need to batch across videos, e.g. first batch = first frame of N videos
    # The problem is that all videos don't have the same frame count.. One hack is to sort videos by frame count
    # and only batch videos of similar frame count together.

    def video_IDs_sorted_by_frame_cnt(self):
        with open(self.args.video_list_file_path, "r") as f:
            video_IDs = [n.strip() for n in f.readlines()]
        frame_cnts = [
            np.load(
                os.path.join(self.args.video_oracle_kpts_folder_path, f"{video_ID}.npy")
            ).shape[0]
            for video_ID in video_IDs
        ]
        video_IDs_frame_cnt = [
            (i, c) for c, i in sorted(zip(frame_cnts, video_IDs), reverse=True)
        ]

        return video_IDs_frame_cnt

    def get_next_chunk_video_IDs(self):
        chunk_video_names = []
        chunk_size = min(self.remaining_videos, self.args.WFLW_V_batch_size)
        chunk_frame_cnt = self.video_IDs_fcnts[self.read_head][
            1
        ]  # all batch should have this frame count

        for _ in range(chunk_size):
            video_name, frame_cnt = self.video_IDs_fcnts[self.read_head]
            if frame_cnt == chunk_frame_cnt:
                chunk_video_names.append(video_name)
                self.read_head += 1
                self.remaining_videos -= 1
            else:
                break

        return chunk_video_names, chunk_frame_cnt

    def get_next_chunk_data(self):
        """This can be quite slow for large batches"""
        video_IDs_chunk, frame_cnt = self.get_next_chunk_video_IDs()
        chunk_size = len(video_IDs_chunk)

        video_faces_chunk = np.zeros(
            (chunk_size, frame_cnt, 256, 256, 3), dtype=np.uint8
        )
        oracle_kpts_chunk = np.zeros((chunk_size, frame_cnt, 98, 2))
        transform_matrices_chunk = np.zeros((chunk_size, frame_cnt, 2, 3))

        for idx, video_ID in enumerate(video_IDs_chunk):
            path_to_video = os.path.join(
                self.args.video_frames_folder_path, f"{video_ID}.mp4"
            )
            assert os.path.isfile(path_to_video)
            video_capture = cv2.VideoCapture(path_to_video)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames = [video_capture.read()[1] for _ in range(n_frames)]
            video_capture.release()

            oracle_kpts_chunk[idx] = np.load(
                os.path.join(self.args.video_oracle_kpts_folder_path, f"{video_ID}.npy")
            )
            bboxes = np.load(
                os.path.join(self.args.video_bboxes_folder_path, f"{video_ID}.npy")
            )
            video_transform_matrices = np.array(
                [
                    get_transform_from_bbox(bbox, extra_scale=1.2, target_im_size=256)
                    for bbox in bboxes
                ]
            )
            transform_matrices_chunk[idx] = video_transform_matrices
            video_faces_np = np.array(
                [
                    cv2.warpAffine(
                        video_frames[i],
                        video_transform_matrices[i],
                        (256, 256),
                        flags=cv2.INTER_LINEAR,
                    )
                    for i in range(n_frames)
                ]
            )  # each face is (256,256,3), whole thing has type unint8
            video_faces_chunk[idx] = video_faces_np

        return (
            video_faces_chunk,
            oracle_kpts_chunk,
            transform_matrices_chunk,
            video_IDs_chunk,
            chunk_size,
            frame_cnt,
        )

    def get_next_chunk_data_parallelized(self):
        """Use CPU parallelization"""
        video_IDs_chunk, frame_cnt = self.get_next_chunk_video_IDs()
        chunk_size = len(video_IDs_chunk)
        # print(f'Processing following videos with {frame_cnt} frames: {video_IDs_chunk}')

        video_faces_chunk = np.zeros(
            (chunk_size, frame_cnt, 256, 256, 3), dtype=np.uint8
        )
        oracle_kpts_chunk = np.zeros((chunk_size, frame_cnt, 98, 2))
        transform_matrices_chunk = np.zeros((chunk_size, frame_cnt, 2, 3))

        self.video_idx = 0
        (
            video_frames_folder_paths,
            video_oracle_kpts_folder_paths,
            video_bboxes_folder_paths,
        ) = (
            [self.args.video_frames_folder_path] * chunk_size,
            [self.args.video_oracle_kpts_folder_path] * chunk_size,
            [self.args.video_bboxes_folder_path] * chunk_size,
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.args.WFLW_V_workers
        ) as executor:
            for (
                video_ID,
                oracle_kpts,
                video_transform_matrices,
                video_faces_np,
            ) in executor.map(
                get_video_data,
                video_IDs_chunk,
                video_frames_folder_paths,
                video_oracle_kpts_folder_paths,
                video_bboxes_folder_paths,
            ):
                idx = video_IDs_chunk.index(video_ID)  # index within chunk of videos
                oracle_kpts_chunk[idx] = oracle_kpts
                transform_matrices_chunk[idx] = video_transform_matrices
                video_faces_chunk[idx] = video_faces_np

            # oracle_kpts_chunk[idx] = oracle_kpts
        # print(results)

        return (
            video_faces_chunk,
            oracle_kpts_chunk,
            transform_matrices_chunk,
            video_IDs_chunk,
            chunk_size,
            frame_cnt,
        )

    @torch.no_grad()
    def test_all_videos_batched(self, RWR):
        """
        NB: to get the same exact results as test_all_videos_sequential, you need to set
        self.train_args.rel_diff_target = 0.0
        """

        self.video_IDs_fcnts = self.video_IDs_sorted_by_frame_cnt()
        self.remaining_videos = len(self.video_IDs_fcnts[:4])
        self.read_head = 0
        NMEs, NMJs = [], []

        self.train_args.verbose_solver = self.args.verbose_solver
        if RWR:
            self.train_args.max_iters = self.args.rwr_max_iters
            self.train_args.take_one_less_inference_step = (
                self.args.rwr_take_one_less_inference_step
            )

        while self.remaining_videos > 0:
            ## -------------- Create chunk
            (
                video_faces_chunk,
                oracle_kpts_chunk,
                transform_matrices_chunk,
                video_IDs_chunk,
                chunk_size,
                frame_cnt,
            ) = self.get_next_chunk_data_parallelized()

            ## -------------- Iterate through chunk. Each frame of chunk is a batch of images
            pred_frame_kpts_chunk = np.zeros((chunk_size, frame_cnt, 98, 2))
            prev_z_star = None
            for batch_idx in range(frame_cnt):
                video_faces_torch = torch.stack(
                    [
                        self.normalize(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                        for face in video_faces_chunk[:, batch_idx, :, :, :]
                    ]
                ).to(self.args.device)

                if RWR:
                    z0 = (
                        prev_z_star
                        if prev_z_star is not None
                        else self.get_z0(chunk_size)
                    )
                    output = self.model(
                        video_faces_torch,
                        mode=self.train_args.model_mode,
                        args=self.train_args,
                        z0=z0,
                    )
                    face_kpts = output["keypoints"].cpu().numpy() * 256
                    frame_kpts = np.stack(
                        [
                            apply_affine_transform_to_kpts(
                                kpts, transform_matrix, inverse=True
                            )
                            for (kpts, transform_matrix) in zip(
                                face_kpts, transform_matrices_chunk[:, batch_idx, :, :]
                            )
                        ]
                    )
                    pred_frame_kpts_chunk[:, batch_idx, :, :] = frame_kpts
                    prev_z_star = output["z_star"]

                else:
                    z0 = self.get_z0(chunk_size)
                    output = self.model(
                        video_faces_torch,
                        mode=self.train_args.model_mode,
                        args=self.train_args,
                        z0=z0,
                    )
                    face_kpts = output["keypoints"].cpu().numpy() * 256
                    frame_kpts = np.stack(
                        [
                            apply_affine_transform_to_kpts(
                                kpts, transform_matrix, inverse=True
                            )
                            for (kpts, transform_matrix) in zip(
                                face_kpts, transform_matrices_chunk[:, batch_idx, :, :]
                            )
                        ]
                    )
                    pred_frame_kpts_chunk[:, batch_idx, :, :] = frame_kpts

            ## -------------- Calculate NME/NMJ for each video in chunk
            for idx, video_ID in enumerate(video_IDs_chunk):
                NME, NMJ = video_NME_NMJ(
                    pred_frame_kpts_chunk[idx], oracle_kpts_chunk[idx]
                )
                print(f"video {video_ID} (NME, NMJ) = ({NME:.2f}, {NMJ:.2f})")
                NMEs.append(NME)
                NMJs.append(NMJ)

        print(f"\n\nAvg (NME, NMJ) = {np.mean(NMEs):02.2f}, {np.mean(NMJs):02.2f}")


def get_video_data(
    video_ID,
    video_frames_folder_path,
    video_oracle_kpts_folder_path,
    video_bboxes_folder_path,
):
    # print('inside', video_ID)
    path_to_video = os.path.join(video_frames_folder_path, f"{video_ID}.mp4")
    assert os.path.isfile(path_to_video)
    video_capture = cv2.VideoCapture(path_to_video)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = [video_capture.read()[1] for _ in range(n_frames)]
    video_capture.release()

    oracle_kpts = np.load(
        os.path.join(video_oracle_kpts_folder_path, f"{video_ID}.npy")
    )
    bboxes = np.load(os.path.join(video_bboxes_folder_path, f"{video_ID}.npy"))
    video_transform_matrices = np.array(
        [
            get_transform_from_bbox(bbox, extra_scale=1.2, target_im_size=256)
            for bbox in bboxes
        ]
    )
    video_faces_np = np.array(
        [
            cv2.warpAffine(
                video_frames[i],
                video_transform_matrices[i],
                (256, 256),
                flags=cv2.INTER_LINEAR,
            )
            for i in range(n_frames)
        ]
    )  # each face is (256,256,3), whole thing has type unint8

    return video_ID, oracle_kpts, video_transform_matrices, video_faces_np


##########################


def main(args):
    if args.device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    t0 = time.time()
    solver = DEQInference(args)
    print(f"Model loaded in {format_time(time.time() - t0)}")

    t0 = time.time()
    solver.test_all_videos_sequential(
        RWR=args.rwr,
        plot=True,
        num_video=args.num_video,
        start_video_idx=args.start_video_idx,
    )
    # solver.test_all_videos_batched(RWR=args.rwr)

    print(f"Total time: {format_time(time.time() - t0)}")

    if args.device == "cuda":
        print(
            f"Max mem: {torch.cuda.max_memory_allocated(device='cuda') / (1024**3):.1f} GB"
        )
    elif args.device == "mps":
        print(f"Max mem: {torch.mps.current_allocated_memory() / (1024**3):.1f} GB")
    else:
        print("Max mem: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEQ Inference")

    parser.add_argument(
        "--landmark_model_weights",
        default="./checkpoints/WFLW_V-final.pth.tar",
        help="path to landmark model weights",
    )
    parser.add_argument(
        "--verbose_solver",
        type=str2bool,
        default=False,
        help="print solver info",
    )
    parser.add_argument(
        "--rwr",
        type=str2bool,
        default=True,
        help="Do recurrence without recurrence",
    )
    parser.add_argument(
        "--rwr_max_iters",
        type=int,
        default=1,
    )  # usually 1
    parser.add_argument(
        "--rwr_take_one_less_inference_step",
        type=str2bool,
        default=False,
    )  # if True, effective n_iters=rwr_max_iters, otherwise n_iters=rwr_max_iters+1
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="cpu, mps or cuda",
    )
    parser.add_argument(
        "--start_video_idx",
        type=int,
        default=0,
        help="idx of the first video",
    )
    parser.add_argument(
        "--num_video",
        type=int,
        default=8,
        help="num of videos to test",
    )
    parser.add_argument(
        "--WFLW_V_split",
        type=str,
        choices=["hard", "easy"],
        default="easy",
    )
    parser.add_argument(
        "--WFLW_V_dataset_path",
        type=str,
        default="../WFLW_V_release",
        help="path to WFLW_V dataset",
    )
    parser.add_argument(
        "--WFLW_V_batch_size",
        type=int,
        default=4,
        help="max num of videos that each have their frame i combined into a single batch",
    )
    parser.add_argument(
        "--WFLW_V_workers",
        type=int,
        default=8,
        help="loading of multiple videos can be slow, so parallelize it across cpu cores",
    )

    args = parser.parse_args()

    print(f"--> args: \n {args}")

    args.video_bboxes_folder_path = os.path.join(args.WFLW_V_dataset_path, "bboxes")
    args.video_oracle_kpts_folder_path = os.path.join(
        args.WFLW_V_dataset_path, "landmarks"
    )
    args.video_frames_folder_path = os.path.join(args.WFLW_V_dataset_path, "videos")
    args.video_list_file_path = f"./datasets/WFLW_V/{args.WFLW_V_split}_video_IDs.txt"
    print("\nStarting...")

    main(args)
