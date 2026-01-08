import numpy as np
import cv2
import av
import random
import tqdm

def stack_frames(fname):
    frames = []
    with av.open(fname) as container:
        for frame in container.decode(video=0): 
            img = frame.to_ndarray(format="rgb24") #convert frame in numpy array 
            frames.append(img)
    return np.stack(frames, axis=0) #(num_frames,H,W,3)


def white_margin_matting(frames):
    height = frames.shape[1]
    width = frames.shape[2]
    margin = (width - height) // 2
    return frames[:, :, margin: margin + height, :]


class ImageSource(object):
    def get_image(self):
        pass

    def reset(self):
        pass


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=1000):
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_array()
        self.current_idx = 0
        self.reset()

    def build_array(self):
        self.arr = np.zeros(
            (self.total_frames, self.shape[0], self.shape[1],3),dtype=np.uint8
        )
        total_frame_i = 0
        file_i = 0
        with tqdm.tqdm(
            total=self.total_frames, desc="Loading videos for natural"
        ) as pbar:
            while total_frame_i < self.total_frames:
                if file_i % len(self.filelist) == 0:
                    random.shuffle(self.filelist)
                file_i += 1
                fname = self.filelist[file_i % len(self.filelist)]
                frames = stack_frames(fname)

                # if (
                #     frames.shape[2] > frames.shape[1] and
                #     frames.shape[0] == 1000 and frames.shape[1] == 100 and
                #     (frames.shape[2] == 240 or frames.shape[2] == 304)
                # ):
                #     frames = white_margin_matting(frames)
                for frame_i in range(frames.shape[0]):
                    if total_frame_i >= self.total_frames:
                        break                       
                    self.arr[total_frame_i] = cv2.resize(
                        frames[frame_i], (self.shape[1], self.shape[0]) # resize the frame to the target (width, height)
                    )
                    pbar.update(1)
                    total_frame_i += 1

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img

