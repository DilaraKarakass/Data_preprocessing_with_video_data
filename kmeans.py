import cv2
import numpy as np
import os

def apply_kmeans_segmentation(image, k=4):
    """Uygulanan K-Means segmentasyon fonksiyonu."""
    # Görüntüyü iki boyutlu matris olarak yeniden boyutlandırma
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means kriterlerini ayarlama
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Renk merkezlerini tamsayıya çevirme ve piksel değerlerini atama
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def process_video_and_save_frames(video_path, output_dir, frame_skip=10, k=4):
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    video_folder = os.path.join(output_dir, video_title)
    os.makedirs(video_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Her frame_skip sayıda karede birini işleme al
        if frame_index % frame_skip == 0:
            # K-Means segmentasyon uygulama
            segmented_frame = apply_kmeans_segmentation(frame, k=k)
            frame_filename = f"{video_title}_frame_{saved_frame_index}_segmented.jpg"
            frame_path = os.path.join(video_folder, frame_filename)
            cv2.imwrite(frame_path, segmented_frame)
            saved_frame_index += 1

        frame_index += 1
    
    cap.release()
    print(f"Processed and saved segmented frames for video: {video_title}")

# Kullanım örneği
video_path = r"C:\Users\Ali\Downloads\Compressed\drive-download-20241101T164948Z-001\giyinmek.mp4" # Video yolunu belirtin
output_dir = r"C:\Users\Ali\Desktop\processed_videos"     # Klasör yolunu belirtin
process_video_and_save_frames(video_path, output_dir, frame_skip=10, k=4)