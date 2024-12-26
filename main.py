import face_recognition
import os
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

# 1) known_images 폴더 내 파일을 자동으로 불러와 Encodings + 폴더 생성
known_images_folder = "known_images"
known_faces = []
known_names = []
output_folders = {}

for known_file in os.listdir(known_images_folder):
    if known_file.lower().endswith((".jpg", ".jpeg", ".png")):
        # 파일명에서 확장자 제거 -> 사람 이름
        name, _ = os.path.splitext(known_file)
        file_path = os.path.join(known_images_folder, known_file)

        # 인코딩 추출
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Warning: No face found for '{known_file}'")
            continue

        # 첫 번째 얼굴 인코딩 사용 (복수 얼굴이면 필요한 로직 추가 가능)
        encoding = encodings[0]
        known_names.append(name)
        known_faces.append(encoding)

        # 사람 이름 기반 output 폴더 준비
        output_folder_name = f"output_{name}"
        os.makedirs(output_folder_name, exist_ok=True)
        output_folders[name] = output_folder_name

if len(known_faces) == 0:
    print("No valid face encodings found in known_images folder. Exiting.")
    exit()

target_images_folder = "datasets"

unknown_threshold = 0.45
output_path_unknown = "output_unknown"
if not os.path.exists(output_path_unknown):
    os.makedirs(output_path_unknown)

output_path_group = "output_group"
os.makedirs(output_path_group, exist_ok=True)

def _find_closest_match(target_image_encoding, known_faces):
    distances = face_recognition.face_distance(known_faces, target_image_encoding)
    closest_match = distances.argmin()
    min_distance = distances[closest_match]
    return closest_match, min_distance

def get_name_by_index(i):
    return known_names[i] if 0 <= i < len(known_names) else "unknown"

for file in os.listdir(target_images_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        target_image_path = os.path.join(target_images_folder, file)
        target_image = face_recognition.load_image_file(target_image_path)
        face_locations = face_recognition.face_locations(target_image)
        target_image_encodings = face_recognition.face_encodings(target_image)

        # 로그 기록 함수
        def write_log(message: str):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"[{now}] {message}\n")

        # 얼굴이 없는 경우 -> Unknown 폴더
        if len(target_image_encodings) == 0:
            print(f"No face found in {file}")
            shutil.copy(target_image_path, os.path.join(output_path_unknown, file))
            write_log(f"'{file}' → no faces found, distance=N/A, copied to '{output_path_unknown}'")
            continue

        # 얼굴이 하나 이상 있는 경우
        pil_image = Image.fromarray(target_image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()

        if len(target_image_encodings) == 1:
            # Single face scenario
            single_encoding = target_image_encodings[0]
            closest_match, min_distance = _find_closest_match(single_encoding, known_faces)

            # 얼굴별 distance 기록
            all_distances = face_recognition.face_distance(known_faces, single_encoding)
            distance_info_list = [f"{known_names[i]}: {all_distances[i]:.4f}" for i in range(len(known_faces))]
            distance_info_str = ", ".join(distance_info_list)
            write_log(f"'{file}' → distances {{ {distance_info_str} }}")

            if min_distance < unknown_threshold:
                matched_person = get_name_by_index(closest_match)
                # 해당 person 이름의 output 폴더 찾기
                if matched_person in output_folders:
                    output_folder = output_folders[matched_person]
                else:
                    matched_person = "unknown"
                    output_folder = output_path_unknown
                    print(f"Unexpected: matched index={closest_match}, but no folder found. Using unknown.")
                    write_log(f"Unexpected no folder for {matched_person}, forced to unknown folder.")
                print(f"Closest match for {file} face 0: {matched_person} (distance={min_distance})")
            else:
                matched_person = "unknown"
                output_folder = output_path_unknown
                print(f"Closest match for {file} face 0: unknown (distance={min_distance})")

            top, right, bottom, left = face_locations[0]
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
            draw.text((left, bottom + 5), matched_person, fill="red", font=font)

        else:
            # Multiple faces scenario
            # 1) Choose main face by largest area
            largest_area = 0
            main_index = 0
            for i, (top, right, bottom, left) in enumerate(face_locations):
                area = (right - left) * (bottom - top)
                if area > largest_area:
                    largest_area = area
                    main_index = i

            # 2) Classify only the main face
            main_encoding = target_image_encodings[main_index]
            closest_match, min_distance = _find_closest_match(main_encoding, known_faces)

            # 여러 얼굴 중 main face distance 기록
            all_distances = face_recognition.face_distance(known_faces, main_encoding)
            distance_info_list = [f"{known_names[i]}: {all_distances[i]:.4f}" for i in range(len(known_faces))]
            distance_info_str = ", ".join(distance_info_list)
            write_log(f"'{file}' → distances {{ {distance_info_str} }}")

            if min_distance < unknown_threshold:
                matched_person = get_name_by_index(closest_match)
                if matched_person in output_folders:
                    output_folder = output_folders[matched_person]
                else:
                    matched_person = "unknown"
                    output_folder = output_path_unknown
                    print(f"Unexpected: matched index={closest_match}, but no folder found. Using unknown.")
                    write_log(f"Unexpected no folder for {matched_person}, forced to unknown folder.")
                print(f"Multiple faces found in {file}, main face recognized as {matched_person} (distance={min_distance})")
            else:
                matched_person = "unknown"
                output_folder = output_path_unknown
                print(f"Multiple faces found in {file}, main face recognized as unknown (distance={min_distance})")

            # 3) Blur all other faces
            for i, (top, right, bottom, left) in enumerate(face_locations):
                if i != main_index:
                    face_region = pil_image.crop((left, top, right, bottom))
                    blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=15))
                    pil_image.paste(blurred_face, (left, top, right, bottom))

            # 4) Draw box/label on main face
            main_top, main_right, main_bottom, main_left = face_locations[main_index]
            draw.rectangle(((main_left, main_top), (main_right, main_bottom)), outline="red", width=2)
            draw.text((main_left, main_bottom + 5), matched_person, fill="red", font=font)

        # Copy original file to the output folder
        shutil.copy(target_image_path, os.path.join(output_folder, file))

        # Save bounding-boxed/blurred image
        boxed_image_path = os.path.join(output_folder, f"boxed_{file}")
        pil_image.save(boxed_image_path)
        print(f"Saved boxed image to {boxed_image_path}")

        # Write log
        write_log(f"'{file}' → classified as '{matched_person}', distance={min_distance}, copied to '{output_folder}'")
