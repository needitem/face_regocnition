import threading
import traceback
import faulthandler
import psutil
import os
import dlib

from engine import FaceRecognitionEngine
from ui import FaceRecognitionUI

def thread_exception_handler(args):
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

def main():
    log_file = open("faulthandler.log", "w", encoding="utf-8")
    faulthandler.enable(file=log_file)
    
    engine = FaceRecognitionEngine()
    ui = FaceRecognitionUI(engine)

    # UI에서 "Start" 버튼 누르면 멀티스레드로 실행하는 로직이 있지만,
    # 임시로 아래처럼 단일 스레드로 직접 테스트해볼 수 있습니다:
    """
    import os
    dataset = "D:/그냥/facerecog/datasets"
    output_base = "D:/그냥/facerecog/output"
    known_dir = "D:/그냥/facerecog/known_images"
    engine.set_threshold(0.45)

    # 단일 스레드 순차 처리:
    # => process_images_in_background 내부 로직을 직접 풀어서 순차 처리
    #    만약 여기서 튕기지 않는다면, 멀티스레드 문제일 가능성이 큼
    known_faces = []
    known_names = []
    for known_file in os.listdir(known_dir):
        if known_file.lower().endswith((".jpg", ".jpeg", ".png")):
            name, _ = os.path.splitext(known_file)
            img_path = os.path.join(known_dir, known_file)
            try:
                known_img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(known_img)
                if encs:
                    known_names.append(name)
                    known_faces.append(encs[0])
            except:
                pass

    output_path_unknown = os.path.join(output_base, "output_unknown")
    os.makedirs(output_path_unknown, exist_ok=True)

    all_files = [f for f in os.listdir(dataset) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    person_counts = {}
    for i, f in enumerate(all_files, start=1):
        file, matched_person, thumb = engine.process_single_image(
            f, dataset, known_faces, known_names, output_base, output_path_unknown
        )
        if matched_person:
            person_counts[matched_person] = person_counts.get(matched_person, 0) + 1
        progress_percent = i / len(all_files) * 100
        print(f"[Test-SingleThread] {f} => {matched_person}, {progress_percent:.2f}%")
        # UI queue에 넘기고 싶다면 engine.results_queue.put(...)

    print("[Test-SingleThread] person_counts=", person_counts)
    """

    # 평소처럼 UI를 띄우려면 아래 run()을 사용:
    ui.run()

if __name__ == "__main__":
    print("DLIB use CUDA:", dlib.DLIB_USE_CUDA)
    threading.excepthook = thread_exception_handler
    main()
