from deep_sort.deep_sort import DeepSort  # 引入 DeepSORT
from deep_sort.utils.parser import get_config  # DeepSORT 配置
import torch

# 初始化 DeepSORT
deepsort_cfg = get_config()
deepsort_cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")  # 配置文件路径
deepsort = DeepSort(
    deepsort_cfg.DEEPSORT.REID_CKPT,
    max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
    max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=deepsort_cfg.DEEPSORT.MAX_AGE,
    n_init=deepsort_cfg.DEEPSORT.N_INIT,
    nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
    use_cuda=torch.cuda.is_available()
)

if mode == "video":
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while True:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成 Image
        image = Image.fromarray(frame_rgb)
        # 调用 YOLO 模型进行检测
        output = yolo.detect_image(image)

        # 提取检测框和置信度
        detected_frame = output[0]  # 检测后的图像
        detections = output[1]  # 检测框和置信度信息

        # 检测框转换为 DeepSORT 格式
        bboxes, confidences, classes = [], [], []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # 转换为 xywh 格式
            confidences.append(conf)
            classes.append(cls)

        bboxes = np.array(bboxes)
        confidences = np.array(confidences)

        # 使用 DeepSORT 进行目标跟踪
        outputs = deepsort.update(bboxes, confidences, frame_rgb)

        # 绘制跟踪框和目标 ID
        for output in outputs:
            x1, y1, x2, y2, track_id = output[:5]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # 计算 FPS
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示和保存视频
        cv2.imshow("video", frame)
        if video_save_path != "":
            out.write(frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:  # 按下 Esc 键退出
            break

    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()
