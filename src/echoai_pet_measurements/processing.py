"""
This file is part of the echoAI-PET-measurements project.
"""
import os
import numpy as np
import cv2
import lz4.frame

class Videoconverter:
    """ Preprocessing functions for echo videos
    min_rate: minimum frame rate
    min_frames: minimum required number of frames
    meta_df: data frame from collect_metadata script
    """
    def __init__(self, max_frame_time_ms, min_frames, meta_df):
        self.max_frame_time = max_frame_time_ms
        self.min_rate = 1/max_frame_time_ms*1e3 if max_frame_time_ms is not None else None
        self.min_frames = min_frames
        self.meta_df = meta_df
        self.min_video_len = min_frames*max_frame_time_ms*1e-3 if max_frame_time_ms is not None else None

    def im_scale(self, im, dx, dy):
        """ convert single images to uint8 and resize by scale factors """
        # We can do other things here: e.g. background subtraction or contrast enhancement
        im_scaled = np.uint8((im - np.amin(im)) / (np.amax(im) - np.amin(im)) * 256)
        # im_scaled_eq = cv2.equalizeHist(im_scaled) # histogram equalization (not needed)
        if (dx is not None) & (dy is not None):
            width = int(np.round(im_scaled.shape[1] * 7.5 * dx))
            height = int(np.round(im_scaled.shape[0] * 7.5 * dy))
            im_resized = cv2.resize(im_scaled, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            im_resized = im_scaled
        return im_resized

    def data2imarray(self, im_data, dx=None, dy=None):
        """
        apply imscale function to np.array
        arg: im_array (frame, height, width)
        returns: im_array (height, width, frame)
        """
        im_data = np.squeeze(im_data)
        im_list = [self.im_scale(im_data[im], dx, dy) for im in range(im_data.shape[0])]
        im_array = np.array(im_list, dtype=np.uint16)
        im_array = np.moveaxis(im_array, 0, -1)
        return im_array

    def subsample_time_index_list(self, frame_time, default_rate, n_frames):
        """
        frame_time: time interval between frames [s]
        default_rate: matching frame rate [fps],
        n_frames: number of frames in the output
        """
        default_times = np.arange(0, n_frames, 1) / default_rate
        times = np.arange(0, default_times[-1] + frame_time, frame_time)
        time_index_list = [np.argmin(np.abs(times - t)) for t in default_times]

        return time_index_list

    def subsample_video(self, image_array, frame_time):
        """
        Select frames that are closest to a constant frame rate
        arg: image_array: np.array() [rows, columns, frame]
        """
        rate = 1 / frame_time
        # Check if the video is long enough
        video_len = image_array.shape[-1] / rate
        subsampled_image_array = np.zeros(1)

        if (self.min_video_len < video_len) & (self.min_rate < rate):
            # print('Video is long enough and the rate is good.')
            # Get the frame index list
            time_index_list = self.subsample_time_index_list(frame_time=frame_time,
                                                             default_rate=self.min_rate,
                                                             n_frames=self.min_frames)
            # Subsample video: skip frames by time index
            subsampled_image_array = image_array[:, :, time_index_list]

        return subsampled_image_array

    def load_video(self, file):
        """ Just load a video file """
        try:
            with lz4.frame.open(file, 'rb') as fp:
                data = np.load(fp)
        except IOError as err:
            print('Cannot open npy file.')
            print(err)
            data = None
        return data

    def process_data(self, data, deltaX, deltaY, frame_time):
        output_array = np.zeros(1)
        error = None
        if (0 < deltaX) & (deltaX < 1) & (0 < deltaY) & (deltaY < 1):
            frame_time *= 1e-3
            rate = 1 / frame_time
            video_len = data.shape[0] / rate
            # If the rate is higher, we need more frames
            if (self.min_video_len < video_len) & (self.min_rate < rate):
                image_array = self.data2imarray(im_data=data, dx=deltaX, dy=deltaY)
                output_array = self.subsample_video(image_array=image_array,
                                                    frame_time=frame_time)
            else:
                if self.min_rate >= rate:
                    print(f'Frame rate is too low: {rate:.2f}s^-1. Skipping.')
                    error = 'frame_rate'
                if self.min_video_len >= video_len:
                    print(f'Video is too short: {video_len:.2f}s. Skipping.')
                    error = 'video_len'
        else:
            print('Meta data invalid for {}. Skipping file.')
            error = 'deltaXY'

        return error, output_array

    def process_video(self, filename):
        meta = self.meta_df[self.meta_df.filename == filename]
        output_array = np.zeros(1)
        error=None
        if meta.shape[0] > 0:
            deltaX = np.abs(meta.deltaX.values[0])
            deltaY = np.abs(meta.deltaY.values[0])
            if (0 < deltaX) & (deltaX < 1) & (0 < deltaY) & (deltaY < 1):
                frame_time = meta.frame_time.values[0] * 1e-3
                rate = 1 / frame_time
                file = os.path.join(meta.dir.values[0], filename)
                try:
                    with lz4.frame.open(file, 'rb') as fp:
                        data = np.load(fp)
                except IOError as err:
                    print('Cannot open npy file.')
                    print(err)
                    error='load'
                else:
                    video_len = data.shape[0] / rate
                    # If the rate is higher, we need more frames
                    if (self.min_video_len < video_len) & (self.min_rate < rate):
                        image_array = self.data2imarray(im_data=data, dx=deltaX, dy=deltaY)
                        output_array = self.subsample_video(image_array=image_array,
                                                            frame_time=frame_time)
                    else:
                        if self.min_rate >= rate:
                            print(f'Frame rate is too low: {rate:.2f}s^-1. Skipping.')
                            error='frame_rate'
                        if self.min_video_len >= video_len:
                            print(f'Video is too short: {video_len:.2f}s. Skipping.')
                            error='video_len'
            else:
                print('Meta data invalid for {}. Skipping'.format(filename))
                error='deltaXY'
        else:
            print('No meta data for {}. Skipping'.format(filename))
        return error, output_array
