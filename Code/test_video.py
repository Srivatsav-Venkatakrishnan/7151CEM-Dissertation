
import cv2
import numpy as np
import os
import openvino as ov
from tqdm import tqdm
import argparse

def process_video(video_path, classification_model_xml):
    core = ov.Core()
    config = {ov.properties.inference_num_threads(): 2, ov.properties.hint.enable_cpu_pinning(): True}
    model = core.read_model(model=classification_model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU", config=config)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if frame_height ==2160 and frame_width==3840:
        # Downsample the image by 50%
        frame_width = frame_width // 2
        frame_height = frame_height // 2
        
    output_video_path = "./output_predicted_video/"
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
        
    out = cv2.VideoWriter(output_video_path+"/"+video_path.split("/")[-1], cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    imageName = []
    imageLabels = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    width = []
    height = []
    prob = []
    labelNewFace = ['money','knife','monedero','pistol','smartphone','tarjeta']

    # outputimage_temp =  "./output_frames/"+video_path.split("/")[-1]+"/"
    # if not os.path.exists(outputimage_temp):
    #     os.makedirs(outputimage_temp)
            
    for row in tqdm(range(frame_count)):
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # Get the original dimensions
        original_height, original_width = image.shape[:2]
        
        if original_height ==2160 and original_width==3840:
            # Downsample the image by 50%
            new_width = original_width // 2
            new_height = original_height // 2
            image = cv2.resize(image, (new_width, new_height))
              
          
        
        img_size = 320
        resized_image = cv2.resize(image, (img_size, img_size)) / 255
        resized_image = resized_image.transpose(2, 0, 1)
        
        reshaped_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        
        im_height, im_width, _ = image.shape
        
        output_numpy = compiled_model(reshaped_image)[0]
        results = output_numpy[0]


    
        for result0 in results:
            boxes = result0[0:4]
            probs = result0[4]
            classes = int(result0[5])

            boxes = boxes / 320
            x1, y1, x2, y2 = np.uint16([boxes[0] * im_width, boxes[1] * im_height, boxes[2] * im_width, boxes[3] * im_height])
            if probs > 0.2 and (labelNewFace[classes] == 'knife' or labelNewFace[classes] == 'money' or labelNewFace[classes]=='smartphone'):
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(image, str(labelNewFace[int(classes)]) + " " + str(probs), (x1, y1), 0, 5e-3 * 200, (0, 255, 0), 3)
                print(labelNewFace[classes])

                imageName.append("frame_" + str(row).zfill(5))
                imageLabels.append(labelNewFace[int(classes)])
                
                width.append(x2 - x1)
                height.append(y2 - y1)
                xmin.append(x1)
                ymin.append(y1)
                xmax.append(x2)
                ymax.append(y2)
                prob.append(probs)
                    
        # cv2.imwrite(outputimage_temp + "new_frame_" + str(row).zfill(5) + ".jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out.write( cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    print("Output video saved in the following path.. ",output_video_path)
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Video Object Detection using OpenVINO')
    parser.add_argument('--video_path', type=str, help='Path to the input video file')
    # parser.add_argument('--output_video_path', type=str, help='Path to the output video file')
    # parser.add_argument('--outputimage_file_path', type=str, help='Path to the output image results')
    parser.add_argument('--object_model_xml', type=str, help='Path to the OpenVINO model XML file')
    # parser.add_argument('labelNewFace', type=str, nargs='+', help='List of class labels')

    args = parser.parse_args()

    process_video(args.video_path, args.object_model_xml)

if __name__ == "__main__":
    main()
