from GershikovHorizonAlg import GershikovE


Gershikov_MED_horizon = GershikovE(dsize=(900, 675))  # this is the size used by the original authors














exit()
src_video_folder = r"D:\My Data\All_Maritime_Datasets\SMD videos after cleaning\Videos\Onboard"
src_gt_folder = r"D:\My Data\All_Maritime_Datasets\SMD videos after cleaning\HorizonGt\Onboard"
dst_video_folder = r"D:\My Data\All_Maritime_Datasets\Benchmarking Results\Gershikov et al\Onboard\Video Results"
dst_quantitative_results_folder = r"D:\My Data\All_Maritime_Datasets\Benchmarking Results\Gershikov et al\Onboard\Quantitative Results"

Gershikov_MED_horizon.evaluate(src_video_folder=src_video_folder, src_gt_folder=src_gt_folder,
                               dst_video_folder=dst_video_folder,
                               dst_quantitative_results_folder=dst_quantitative_results_folder, draw_and_save=True)

