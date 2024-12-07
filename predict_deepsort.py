#-----------------------------------------------------------------------#
#   predict_deepsort.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
import torch

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = True
    count           = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 'vdo.avi'
    # video_save_path = "Result_original.avi"
    video_save_path = "Result_new.avi"

    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    confidence      = 0.7
    nms_iou         = 0.5
    model_path      = "logs/ep010-loss0.052-val_loss0.038.pth"
    # model_path      = "logs/last_epoch_weights.pth"
    # model_path      = "model_data/yolov7_weights.pth"


    phi             = "l"
    classes_path    = 'model_data/vehicle_classes.txt'
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_save_path = f'ResultWithoutFooddebris/predictresult_{confidence}'
    dir_origin_path = f'dataset_withoutdebris/testall'
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou, model_path = model_path, phi = phi, classes_path = classes_path)
    else:
        yolo = YOLO_ONNX()

    # 定义颜色映射 (R, G, B 格式)
    class_colors = {
        0: (255, 0, 0),  # 类别 0 使用红色
        1: (0, 255, 0)   # 类别 1 使用绿色
    }
    if mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        # 初始化 DeepSORT
        deepsort_cfg = get_config()
        deepsort_cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        deepsort = DeepSort(
            deepsort_cfg.DEEPSORT.REID_CKPT,
            max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=deepsort_cfg.DEEPSORT.MAX_AGE,
            n_init=deepsort_cfg.DEEPSORT.N_INIT,
            nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
            use_cuda=torch.cuda.is_available()
        )

        fps = 0.0
        while True:
            t1 = time.time()
            ret, frame = capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # 获取 YOLO 检测结果
            output = yolo.detect_image(image)
            #detected_frame = output[0]
            detections = output[1]  # detections 是一个列表，每个元素是 [x1, y1, x2, y2, score, cls]

            # 转换检测框为 DeepSORT 格式
            bboxes, confidences, classes = [], [], []
            if len(detections) > 0:
                for det in detections:
                    if len(det) >= 6:
                        top, left, bottom, right, conf, cls = det[:6]
                        top     = max(0, np.floor(top).astype('int32'))
                        left    = max(0, np.floor(left).astype('int32'))
                        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                        right   = min(image.size[0], np.floor(right).astype('int32'))
                        # 计算宽度和高度
                        w = right - left
                        h = bottom - top
                        # 计算中心坐标
                        x_center = left + w / 2
                        y_center = top + h / 2
                        # 构建边界框信息
                        bboxes.append([x_center, y_center, w, h])
                        confidences.append(conf)
                        classes.append(cls)
                        # 打印调试信息
                        print(f"Detection: top={top}, left={left}, h={h}, w={w}, conf={conf}, cls={cls}")
                    else:
                        print(f"Invalid detection format: {det}")

            else:
                print("No detections found.")
                # 如果没有检测到目标，直接返回空结果
                bboxes = np.empty((0, 4))        # 空检测框
                confidences = np.empty((0,))    # 空置信度
                classes = np.empty((0,))        # 空类别

            # 转换为 NumPy 数组
            bboxes = np.array(bboxes)
            confidences = np.array(confidences)
            classes = np.array(classes)

            # DeepSORT 跟踪
            outputs, _ = deepsort.update(bboxes, confidences, classes, frame)
            # 如果没有跟踪结果，跳过绘制
            if outputs is None or len(outputs) == 0:
                print("No tracking outputs in this frame.")
                outputs = np.empty((0, 6))  # 确保输出为空数组，防止后续处理报错
            
            # 绘制检测框和跟踪 ID
            if outputs is not None and len(outputs) > 0:
                if outputs.ndim == 1:
                    outputs = outputs[np.newaxis, :]
                for output in outputs:
                    print("length+++++", len(output))
                    if len(output) >= 6:
                        x1, y1, x2, y2, track_cls, track_id = output[:6]
                        
                        x1 = int(round(x1))
                        y1 = int(round(y1))
                        x2 = int(round(x2))
                        y2 = int(round(y2))
                        track_id = int(track_id)
                        track_cls = int(track_cls)

                        # 根据类别选择颜色
                        color = class_colors.get(track_cls, (255, 255, 255))  # 如果未定义类别颜色，默认为白色

                        # 绘制跟踪框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # 绘制跟踪 ID 和类别
                        cv2.putText(frame, f"ID: {track_id} Class: {track_cls}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    else:
                        print(f"Invalid output format: {output}")
            else:
                print("No tracking outputs in this frame.")


            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            if video_save_path != "":
                out.write(frame)

            c = cv2.waitKey(1) & 0xff
            if c == 27:
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
