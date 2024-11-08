import os
import torch
import cv2
import numpy as np

# Modeli yükle (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Renk paleti oluştur
def create_color_palette(num_classes):
    np.random.seed(42)  # Renklerin tutarlılığı için sabit rastgelelik
    return np.random.randint(0, 255, size=(num_classes, 3))

# Klasör adı oluşturma
def create_output_folder(base_name, output_root):
    folder_name = os.path.join(output_root, base_name)
    count = 1
    while os.path.exists(folder_name):
        folder_name = os.path.join(output_root, f"{base_name}{count}")
        count += 1
    os.makedirs(folder_name, exist_ok=True)  # Klasörü oluştur
    return folder_name

# Video işleme ve her kareyi kaydetme
def process_video(video_path, output_root):
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Video adını al
    output_folder = create_output_folder(video_name, output_root)  # Çıktı klasörü oluştur

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video file {video_path} could not be opened.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Renk paletini oluştur
    color_palette = create_color_palette(80)  # COCO veri setinde 80 sınıf var
    frame_count = 0  # Kare sayacını başlat

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break

        # Modeli çalıştır
        results = model(frame)

        # Sonuçları al
        detections = results.pred[0]  # 0. katmandaki sonuçları al

        # Görüntüyü görselleştirmek için bir maske oluştur
        for *xyxy, conf, cls in detections:  # xyxy: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)
            color = color_palette[int(cls)]  # Sınıfa göre rengi al
            
            # Nesne kutusunu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)  # Nesne kutusu

            # Maske oluştur ve renklendir
            mask = np.zeros_like(frame)
            mask[y1:y2, x1:x2] = color  # Maske alanını renklendir

            # Görüntü ve maskeyi birleştir
            blended = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)
            frame = blended

        # İşlenen kareyi kaydet
        output_frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")  # 4 haneli numara
        cv2.imwrite(output_frame_path, frame)
        frame_count += 1  # Kare sayacını artır
        print(f"Saved frame {frame_count} to {output_frame_path}")

    cap.release()
    print("Video processing complete.")

# Video dosya yolları
video_path = r"C:\Users\Ali\Downloads\Compressed\drive-download-20241101T164948Z-001\video\dansetmek.mp4" # İşlenecek video dosyanız
output_root = r"C:\Users\Ali\Desktop\instance"  # Çıktı dosyalarının kaydedileceği klasör
process_video(video_path, output_root)