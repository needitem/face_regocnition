import face_recognition
import os
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

person1 = face_recognition.load_image_file("known_images/124.jpeg")
person2 = face_recognition.load_image_file("known_images/113.jpeg")
person3 = face_recognition.load_image_file("known_images/312.jpeg")
person4 = face_recognition.load_image_file("known_images/11.jpeg")

try:
    person1_face_encoding = face_recognition.face_encodings(person1)[0]
    person2_face_encoding = face_recognition.face_encodings(person2)[0]
    person3_face_encoding = face_recognition.face_encodings(person3)[0]
    person4_face_encoding = face_recognition.face_encodings(person4)[0]
except IndexError:
    print("I wasn't able to locate any faces in the photo")
    exit()

known_faces = [person1_face_encoding, person2_face_encoding, person3_face_encoding, person4_face_encoding]

target_images_folder = "datasets"

# Create separate output folders for each person
output_path_person1 = "output_person1"
output_path_person2 = "output_person2"
output_path_person3 = "output_person3"
output_path_person4 = "output_person4"


os.makedirs(output_path_person1, exist_ok=True)
os.makedirs(output_path_person2, exist_ok=True)
os.makedirs(output_path_person3, exist_ok=True)
os.makedirs(output_path_person4, exist_ok=True)

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

            # 로그에 각 얼굴별 distance 기록
            all_distances = face_recognition.face_distance(known_faces, single_encoding)
            name_map = {0: "person1", 1: "person2", 2: "person3", 3: "person4"}
            distance_info_list = [f"{name_map[i]}: {all_distances[i]:.4f}" for i in range(len(known_faces))]
            distance_info_str = ", ".join(distance_info_list)
            write_log(f"'{file}' → distances {{ {distance_info_str} }}")

            if min_distance < unknown_threshold:
                matched_person = (
                    "person1" if closest_match == 0 else
                    ("person2" if closest_match == 1 else
                     ("person3" if closest_match == 2 else "person4"))
                )
                output_folder = (
                    output_path_person1 if closest_match == 0 else
                    (output_path_person2 if closest_match == 1 else
                     (output_path_person3 if closest_match == 2 else output_path_person4))
                )
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

            # 로그에 각 얼굴별 distance 기록
            all_distances = face_recognition.face_distance(known_faces, main_encoding)
            name_map = {0: "person1", 1: "person2", 2: "person3", 3: "person4"}
            distance_info_list = [f"{name_map[i]}: {all_distances[i]:.4f}" for i in range(len(known_faces))]
            distance_info_str = ", ".join(distance_info_list)
            write_log(f"'{file}' → distances {{ {distance_info_str} }}")

            if min_distance < unknown_threshold:
                matched_person = (
                    "person1" if closest_match == 0 else
                    ("person2" if closest_match == 1 else
                     ("person3" if closest_match == 2 else "person4"))
                )
                output_folder = (
                    output_path_person1 if closest_match == 0 else
                    (output_path_person2 if closest_match == 1 else
                     (output_path_person3 if closest_match == 2 else output_path_person4))
                )
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
