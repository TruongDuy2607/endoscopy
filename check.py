import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import os
from PIL import Image
from models.MobileNet_Hierar import HierarchicalMobileNetV3
from timeit import default_timer as timer

torch.cuda.empty_cache()


def main():
    global sub_path, subMini_path, display_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--module_weights', nargs='+', type=str, default='./weights/MBNet_Adam_Focal_3.pt',
                        help='direction to binary classification model: ex resnet50.pt')
    parser.add_argument('--frame_rate', type=int, default=10, help='frame_rate to read each frame from video')
    parser.add_argument('--source', type=str,
                        default=r"D:\Lab-Tracks\Gastrointestinal\Evaluate\VIEM_4_CROP.mp4",
                        help='ex: ./Video/vid3.mp4 or 0 if using source is camera')
    parser.add_argument('--save_vid', action='store_true', default=False, help='save video to the folder')
    parser.add_argument('--show_vid', action='store_true', default=True, help='show video to the monitor')
    parser.add_argument('--save_image', action='store_true', default=False, help='save frame to the folder')
    parser.add_argument('--make_folder', action='store_true', default=False)
    parser.add_argument('--move_frame', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--simulator', action='store_true', default=True, help='run with simulator')
    parser.add_argument('--markov_chain', action='store_true', default=False, help='run with markov chain')
    args = parser.parse_args()
    # source = str(source)
    device = args.device
    class_names = ['Khong xac dinh', 'Mo', 'Bot/Dich', 'Toi', 'Hau hong', 'Thuc quan', 'Tam vi', 'Than vi',
                   'Phinh vi', 'Hang vi', 'Bo cong lon',
                   'Bo cong nho', 'Hanh ta trang', 'Ta trang']

    path_img = r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\Non_color.png"
    img = img_with_mask = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HauHong.png",
                      cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\ThucQuan.png",
                      cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\TamVi.png",
                      cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\ThanVi.png",
                      cv2.IMREAD_GRAYSCALE)
    img5 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\PhinhVi.png",
                      cv2.IMREAD_GRAYSCALE)
    img6 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HangVi.png",
                      cv2.IMREAD_GRAYSCALE)
    img7 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\BoCongLon.png",
                      cv2.IMREAD_GRAYSCALE)
    img8 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\BoCongNho.png",
                      cv2.IMREAD_GRAYSCALE)
    img9 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HanhTaTrang.png",
                      cv2.IMREAD_GRAYSCALE)
    img10 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\TaTrang.png",
                       cv2.IMREAD_GRAYSCALE)

    img1_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HauHong_1.png")
    img2_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\ThucQuan_1.png")
    img3_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\TamVi_1.png")
    img4_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\ThanVi_1.png")
    img5_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\PhinhVi_1.png")
    img6_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HangVi_1.png")
    img7_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\BoCongLon_1.png")
    img8_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\BoCongNho_1.png")
    img9_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\HanhTaTrang_1.png")
    img10_ = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Ver3\TaTrang_1.png")

    ins1 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Hauhong.png")
    ins2 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Thucquan.png")
    ins3 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Tamvi.png")
    ins4 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Thanvi.png")
    ins5 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Phinhvi.png")
    ins6 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Hangvi.png")
    ins7 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Boconglon.png")
    ins8 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Bocongnho.png")
    ins9 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Hanhtatrang.png")
    ins10 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Tatrang.png")
    ins11 = cv2.imread(r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\image\Art\Text_insert\Non.png")


    list_img = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
    list_img_mask = [img1_, img2_, img3_, img4_, img5_, img6_, img7_, img8_, img9_, img10_]
    list_ins = [ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8, ins9, ins10, ins11]
    ins_default = list_ins[10]

    perent_class = ['Non-informative', 'Informative']
    
    x = 0.6
    markov_matrix = torch.tensor([
        [x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x, (1-x) / 9],
        [(1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, (1-x) / 9, x]
    ])
    # markov_matrix = torch.tensor([
    #     [0.8, 0.1, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005],
    #     [0.01, 0.8, 0.1, 0.01, 0.01, 0.01, 0.04, 0.01, 0.005, 0.005],
    #     [0.04, 0.06, 0.8, 0.06, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005],
    # ])

    count_tt = 0
    now_prediction = 0
    pre_prediction = 0
    pre_markov = None
    path_model = args.module_weights
    model = HierarchicalMobileNetV3().to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    time_start = timer()
    # Define transforms to preprocess the frames
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    path_output_save_video = r"D:\Lab-Tracks\Gastrointestinal\GraduationProject\run\result_videos\VIEM_4.mp4"
    out = cv2.VideoWriter(path_output_save_video, 0x7634706d, 30, (1440, 720))

    # Tern for save image
    if args.save_image:
        args.make_folder = True
        args.move_frame = True

    if args.make_folder:
        path_output_save_image = r'D:\Lab-Tracks\Gastrointestinal\GraduationProject\run\result_image'
        sub_path = os.path.join(path_output_save_image, f'run{len(os.listdir(path_output_save_image)) + 1}')
        os.makedirs(sub_path)
        for i in range(2):
            mini_path = os.path.join(sub_path, str(perent_class[i]))
            os.makedirs(mini_path)
            for j in range(14):
                subMini_path = os.path.join(mini_path, str(class_names[j]))
                os.makedirs(subMini_path)
    # Load the video
    video_path = args.source
    cap = cv2.VideoCapture(video_path)
    # Process each frame of the video
    frame_rate = 30 / args.frame_rate  # if frame_rate = 6, quality = 30frame/second -> Each second we'll take 5 frame
    frame_count = 0
    count = 1
    x_offset = 820
    y_offset = 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width, _ = frame.shape
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', frame_width, frame_height)

        # Process every 5th frame
        # print(f'frame_count: {frame_count}')
        if frame_count % frame_rate == 1:
            start_time = timer()
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_tensor_1 = transform(frame_pil).unsqueeze(0)
            frame_tensor_1 = frame_tensor_1.to(device)

            with torch.no_grad():
                output1, output2 = model(frame_tensor_1.to(device))
                output1 = torch.argmax(torch.softmax(output1, dim=1), dim=1).item()

                #If not using Markov chain
                output2 = torch.softmax(output2, dim=1)
                real_output2 = output2.max()
                output2 = torch.argmax(output2, dim=1).item()

                ##If using Markov chain
                # output2 = torch.softmax(output2, dim=1)
                # real_output2 = output2.max() # Confodence score
                # if torch.argmax(output2, dim=1).item() in range(4):
                #     output2 = torch.argmax(output2, dim=1).item()
                #     pre_markov = None
                #     # pre_prediction = None
                # else:
                #     output2 = output2[:, 4:14]
                #     if pre_markov == None:
                #         pre_markov = output2.to(device)
                #         print(pre_markov.shape)
                #         _output2 = torch.argmax(output2, dim=1).item()
                #         output2 = _output2 + 4
                #         # pre_prediction = output2

                #     else:
                #         # pre_markov = output2.to(device)
                #         index_matrix = markov_matrix[torch.argmax(output2, dim=1).item()].to(device)
                #         _output2 = torch.argmax(pre_markov * index_matrix)
                #         pre_markov = _output2.to(device)
                #         _output2 = _output2.item()
                #         output2 = _output2 + 4

            if args.simulator:
                now_prediction = output2 # Assign output2 to current prediction
                if now_prediction == pre_prediction and output2 in range(4, 14):
                    count_tt += 1
                    if (count_tt >= 2):
                        img = cv2.bitwise_or(img, list_img[output2 - 4])
                        mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        img_with_mask = cv2.addWeighted(list_img_mask[output2 - 4], 0.8, mask, 0.2, 0.0)
                        ins_default = list_ins[output2 - 4]
                else:
                    count_tt = 0
                    ins_default = list_ins[10]
                pre_prediction = now_prediction
            if args.make_folder:
                output_class_1 = os.path.join(sub_path, str(perent_class[output1]))
                output_class_2 = os.path.join(output_class_1, str(class_names[output2]))
                image_saved_path = os.path.join(output_class_2, f'mica_{count}.jpg')
                cv2.imwrite(image_saved_path, frame)
            end_time = timer()
            print(f'frame {count}: {class_names[output2]} | {real_output2:.2f}')
            # cv2.putText(frame, f'{class_names[output2]}', (30, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6) if output2 in range(4, 14) else cv2.putText(frame, f'{class_names[output2]}', (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)


        if frame.shape[0] != img_with_mask.shape[0]:
            display_image = cv2.resize(img_with_mask, (img_with_mask.shape[1], frame.shape[0]))
        if len(img_with_mask.shape) == 2 or img_with_mask.shape[2] == 1:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        # Concatenate images horizontally
        combined_image = np.hstack((frame, display_image))
        combined_image = cv2.resize(combined_image, (1440, 720))
        combined_image[y_offset:y_offset+ins_default.shape[0], x_offset:x_offset+ins_default.shape[1]] = ins_default
        if args.save_vid:
            out.write(combined_image)

        if args.show_vid:
            cv2.imshow("Video and Simulator", combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()


        # if args.save_vid:
        #     out.write(frame)
        frame_count += 1
        count += 1

    time_end = timer()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total time to process data: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
