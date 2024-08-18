import cv2 as cv
import numpy as np
from numpy import nan
from time import time
from skimage.filters import median
from math import pi, atan
import os


class GershikovE:
    """
    A class implementing the horizon detection algorithm named H-MED, published by Gershikov et al. in the paper titled:
    'Horizon line detection in marine images: which method to choose?'
    """

    def __init__(self, dsize=None, ekernel=None, eiter=1):
        """
        :param dsize: a tuple (width, height) indicating the resolution to which the image to process will be resized.
        If not given, the original image is processed without changing its resolution.
        :param ekernel: a circular kernel used to preprocess the image using one iteration of morphological erosion.
        :param eiter: number of erosion iterations; defaults to 1.
        """
        # hyper parameters
        self.eiter = eiter
        self.dsize = dsize
        if ekernel is None:
            self.ekernel = np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]],
                                    dtype=np.uint8)  # a circular kernel with radius = 2. It is used for morphological
            # erosion in the preprocessing step: see section 2, D, step 1
        elif isinstance(ekernel, np.ndarray):
            self.ekernel = ekernel
        else:
            raise Exception('the argument ekernel must be a numpy array')

        # images
        self.img_color = None  # input colored image
        self.img_gray = None  # image to process; it is grayscale and resized to self.dsize (if it is not None)
        self.img_with_hl = None  # the color image depicting the found horizon line
        self.eroded_img = None  # self.img_gray after erosion
        self.horizon_edge_map = None  # a binary image containing candidate horizon edges
        self.med_up = None  # self.img_gray median-filtered with self.med_up_kernel
        self.med_down = None  # self.img_gray median-filtered with self.med_down_kernel

        # coordinates
        self.horizon_edges_x = None  # x coordinates of horizon edges
        self.horizon_edges_y = None  # y cooridnates of horizon edges
        self.horizon_edges_xy = None  # a 2D array of two columns:
        # 1st = self.horizon_edges_x, 2nd = self.horizon_edges_y

        # outputs
        self.det_position_hl = nan  # in pixels
        self.det_tilt_hl = nan  # in radians
        self.latency = nan  # in seconds
        self.img_with_hl = None  # the output image with the horizon line
        self.org_height, self.org_width = None, None  # original width and height of the image
        self.resized_height, self.resized_width = None, None  # resized width and height of the image

        # flags
        self.detected_hl_flag = True
        self.med_up_kernel = np.array([[0],
                                       [1],
                                       [1],
                                       [1],
                                       [1],
                                       [1],
                                       [0],
                                       [0],
                                       [0],
                                       [0],
                                       [0]], dtype=np.uint8)  # at the p-th pixel, we used this kernel to compute the
        # median of the 5 pixels above the p-th pixel (including the p-th pixel)

        self.med_down_kernel = np.array([[0],
                                         [0],
                                         [0],
                                         [0],
                                         [0],
                                         [0],
                                         [1],
                                         [1],
                                         [1],
                                         [1],
                                         [1]], dtype=np.uint8)  # at the p-th pixel, we used this kernel to compute the
        # median of the 5 pixels below the p-th pixel (excluding the p-th pixel)

    def get_horizon(self, img):
        self.start_time = time()
        self.img_color = img
        self.img_gray = cv.cvtColor(self.img_color, cv.COLOR_BGR2GRAY)
        self.org_height, self.org_width = self.img_gray.shape
        if self.dsize is not None:
            self.resized_width, self.resized_height = self.dsize
            self.img_gray = cv.resize(src=self.img_gray, dsize=self.dsize)
        else:
            self.resized_width, self.resized_height = self.org_width, self.org_height
        self.preprocess()
        self.get_edges()
        self.fit_horizon()
        self.end_time = time()
        self.latency = round((self.end_time - self.start_time), 4)

    def preprocess(self):
        """
        Applies morphological erosion on the grayscale image self.img_gray with the kernel self.ekernel self.eiter times
        """
        for it in range(1, self.eiter + 1):
            self.eroded_img = cv.erode(src=self.img_gray, kernel=self.ekernel, iterations=it)

    def get_edges(self):
        self.med_up = np.float32(median(image=self.eroded_img, selem=self.med_up_kernel))
        self.med_down = np.float32(median(image=self.eroded_img, selem=self.med_down_kernel))
        self.edge_response = np.uint8(np.abs(np.subtract(self.med_up, self.med_down)))

        self.horizon_edge_map = np.zeros(shape=self.edge_response.shape, dtype=np.uint8)

        self.horizon_edges_y = np.argmax(self.edge_response,
                                         axis=0)  # get y coordinates of horizon edges, which are the
        # maximum values along the axis 0 (the y-axis). The number of such coordinates is equal to the image columns,
        # i.e., width of the image.
        self.horizon_edges_x = np.arange(0, self.resized_width)  # x coordinates of horizon edges; one edge pixel/column
        self.horizon_edges_xy = np.zeros((self.horizon_edges_x.size, 2), dtype=np.int32)
        self.horizon_edges_xy[:, 0], self.horizon_edges_xy[:, 1] = self.horizon_edges_x, self.horizon_edges_y
        self.horizon_edge_map[self.horizon_edges_y, self.horizon_edges_x] = 255

    def fit_horizon(self):
        [self.vx, self.vy, self.x, self.y] = cv.fitLine(points=self.horizon_edges_xy, distType=cv.DIST_L2,
                                                        param=0, reps=1, aeps=pi / 180)
        self.hl_slope = float(self.vy / self.vx)  # float to convert from (1,) float numpy array to python float
        self.height_ratio = self.org_height / self.resized_height
        self.hl_intercept = float(self.y - self.hl_slope * self.x) * self.height_ratio  # for the original size

        self.xs_hl = int(0)
        self.xe_hl = int(self.org_width - 1)
        self.ys_hl = int(self.hl_intercept)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
        self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)

        self.det_tilt_hl = (-atan(self.hl_slope)) * (180 / pi)  # - because the y axis of images goes down
        self.det_position_hl = ((self.org_width - 1) / 2) * self.hl_slope + self.hl_intercept

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl' if it is detected. Otherwise, the text 'NO HORIZON IS
        DETECTED' is put on the image.
        """
        self.img_with_hl = np.copy(self.img_color)
        if self.detected_hl_flag:
            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255), 5)
        else:
            put_text = "NO HORIZON IS DETECTED"
            org = (int(self.org_height / 2), int(self.org_width / 2))
            color = (0, 0, 255)
            cv.putText(img=self.img_with_hl, text=put_text, org=org, fontFace=0, fontScale=2, color=color, thickness=3)
        # points = np.ones(shape=(self.resized_height, self.resized_width, 3), dtype=np.uint8) * 255
        # for x, y in zip(self.horizon_edges_xy[:, 0], self.horizon_edges_xy[:, 1]):
        #     center = (x, y)
        #     cv.circle(img=points, center=center, radius=2, color=(0, 0, 255), thickness=2)
        #     print("circile drawn")
        # cv.imwrite("horizon points.png", points)

    def evaluate_old(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                     draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For
        instance, if the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform
        is sqrt(640^2 + 480^2) * hough_threshold_ratio, rounded to the nearest integer. :param src_gt_folder:
        absolute path to the ground truth horizons corresponding to source video files. :param src_video_folder:
        absolute path to folder containing source video files to process :param dst_video_folder: absolute path where
        video files with drawn horizon will be saved. :param dst_quantitative_results_folder: destination folder
        where quantitative results will be saved. :param draw_and_save: if True, all detected horizons will be drawn
        on their corresponding frames and saved as video files in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
                       " to proceed, otherwise, click on 'n'.\n"
                       "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
                       "files with similar names.")
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "E.Ger_" + src_video_name)
                video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                              True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                error_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                raise Exception(error_text_1)

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.get_horizon(img=frame)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon
                print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                      format(round(self.det_position_hl, 2), round(self.gt_position_hl, 2), round(self.det_tilt_hl, 2),
                             round(self.gt_tilt_hl, 2)))
                print("with latency = {} seconds".format(round(self.latency, 4)))
                self.det_horizons_per_file[idx] = [self.det_position_hl,
                                                   self.det_tilt_hl,
                                                   round(abs(self.det_position_hl - self.gt_position_hl), 4),
                                                   round(abs(self.det_tilt_hl - self.gt_tilt_hl), 4),
                                                   self.latency]
                self.draw_hl()  # draws the horizon on self.img_with_hl
                video_writer.write(self.img_with_hl)
            cap.release()
            video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For
        instance, if the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform
        is sqrt(640^2 + 480^2) * hough_threshold_ratio, rounded to the nearest integer. :param src_gt_folder:
        absolute path to the ground truth horizons corresponding to source video files. :param src_video_folder:
        absolute path to folder containing source video files to process :param dst_video_folder: absolute path where
        video files with drawn horizon will be saved. :param dst_quantitative_results_folder: destination folder
        where quantitative results will be saved. :param draw_and_save: if True, all detected horizons will be drawn
        on their corresponding frames and saved as video files in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
                       " to proceed, otherwise, click on 'n'.\n"
                       "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
                       "files with similar names.")
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "Gershikov_" + src_video_name)
                video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                              True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                warning_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print("----------WARNING---------")
                print(warning_text_1)
                print("--------------------------")

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.get_horizon(img=frame)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                # print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                #       format(round(self.det_position_hl, 2), round(self.gt_position_hl, 2), round(self.det_tilt_hl, 2),
                #              round(self.gt_tilt_hl, 2)))
                # print("with latency = {} seconds".format(round(self.latency, 4)))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.det_position_hl,
                                                   self.det_tilt_hl,
                                                   round(abs(self.det_position_hl - self.gt_position_hl), 4),
                                                   round(abs(self.det_tilt_hl - self.gt_tilt_hl), 4),
                                                   self.latency]
                self.draw_hl()  # draws the horizon on self.img_with_hl
                video_writer.write(self.img_with_hl)
            cap.release()
            video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)