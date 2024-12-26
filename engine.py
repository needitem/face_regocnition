import os
import shutil
import queue
import face_recognition
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk
import traceback
import logging
import psutil
import numpy as np

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler = logging.FileHandler('log.txt', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class FaceRecognitionEngine:
    def __init__(self):
        logger.info("Initializing FaceRecognitionEngine")
        self.unknown_threshold = 0.45
        self.results_queue = queue.Queue()

    def set_threshold(self, value: float):
        logger.info(f"[set_threshold] Setting threshold to {value}")
        self.unknown_threshold = value

    def _find_closest_match(self, target_encoding, known_faces):
        logger.info(f"[_find_closest_match] Called with {len(known_faces)} known faces.")
        try:
            distances = face_recognition.face_distance(known_faces, target_encoding)
            min_idx = distances.argmin()
            logger.debug(f"[_find_closest_match] Found min_idx={min_idx}, distance={distances[min_idx]}")
            return min_idx, distances[min_idx]
        except Exception as e:
            logger.exception("Error in _find_closest_match")
            return 0, 999.0  # 임의로 큰 거리 반환

    def _log_memory_usage(self, prefix: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024
        logger.info(f"[MEMORY] {prefix} RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")

    def process_single_image(self, file, dataset_folder, known_faces, known_names,
                             base_output_folder, output_path_unknown):
        self._log_memory_usage("Before processing single image")
        try:
            file_path = os.path.join(dataset_folder, file)

            # ----------------------------------------------------
            # 1) PIL로 먼저 이미지 파일이 손상되지 않았는지 검사
            # ----------------------------------------------------
            try:
                with Image.open(file_path) as test_img:
                    test_img.load()  # 실제 로딩하여 손상 여부 체크
                logger.debug(f"[process_single_image] PIL check passed for {file}")
            except Exception as e:
                logger.exception(f"[process_single_image] Image appears corrupted: {file}")
                # 손상된 파일 처리 (unknown 폴더 복사 또는 무시)
                corrupted_path = os.path.join(output_path_unknown, "corrupted_files")
                os.makedirs(corrupted_path, exist_ok=True)
                shutil.copy(file_path, os.path.join(corrupted_path, file))
                return file, "unknown", None

            # ----------------------------------------------------------------
            # (추가) PIL로 원본 이미지를 열어서 리사이즈한 뒤에
            # face_recognition이 처리하도록
            # ----------------------------------------------------------------
            try:
                with Image.open(file_path) as im:
                    # 예: 가로세로 최대 800 px로 축소
                    max_width, max_height = 800, 800
                    im.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    # face_recognition에서 numpy 배열 형태를 필요로 하므로
                    # im을 numpy 배열로 변환
                    image_array = np.array(im)
            except Exception as e:
                logger.exception(f"[process_single_image] Reopen & resize error: {file}")
                return file, "unknown", None

            # 이제 face_recognition.load_image_file() 대신
            # 바로 image_array를 넘김
            # face_recognition.face_locations() 등도 array를 직접 처리 가능
            face_locations = face_recognition.face_locations(image_array)
            encodings = face_recognition.face_encodings(image_array)

            logger.debug(f"[process_single_image] face_locations: {face_locations}")
            logger.debug(f"[process_single_image] encodings found: {len(encodings)}")

            pil_image = Image.fromarray(image_array)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()

            matched_person = "unknown"
            if not encodings:
                logger.info("[process_single_image] No face detected, copying to unknown folder.")
                shutil.copy(file_path, os.path.join(output_path_unknown, file))
            elif len(encodings) == 1:
                logger.info("[process_single_image] Exactly one face detected.")
                idx, dist = self._find_closest_match(encodings[0], known_faces)
                logger.info(f"[process_single_image] Closest match: {known_names[idx] if idx < len(known_names) else '??'} / dist={dist}")
                if dist < self.unknown_threshold and idx < len(known_names):
                    matched_person = known_names[idx]
                top, right, bottom, left = face_locations[0]
                draw.rectangle(((left, top), (right, bottom)), outline="red", width=5)
                draw.text((left, bottom + 5), matched_person, fill="red", font=font)
            else:
                logger.info(f"[process_single_image] Multiple faces detected: {len(encodings)}")
                largest_area = 0
                main_index = 0
                face_areas = []
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    area = (right - left) * (bottom - top)
                    face_areas.append(area)
                    if area > largest_area:
                        largest_area = area
                        main_index = i

                idx, dist = self._find_closest_match(encodings[main_index], known_faces)
                logger.info(f"[process_single_image] Main face match: {idx} / dist={dist}")
                if dist < self.unknown_threshold and idx < len(known_names):
                    matched_person = known_names[idx]

                # ---------------------------------------------------------
                # 1) 여럿 얼굴 중 "두 번째로 큰 얼굴"의 면적이
                #    가장 큰 얼굴의 area * R(예: 0.3) 이하인지 확인
                #    => 이 비율보다 작다면 '압도적으로 큰 얼굴 한 명'
                #       으로 간주하고, 사실상 단일 얼굴처럼 처리
                # 2) 그렇지 않다면, 기존대로 '여러 얼굴'
                # ---------------------------------------------------------
                sorted_areas = sorted(face_areas, reverse=True)
                second_largest = sorted_areas[1] if len(sorted_areas) > 1 else 0
                ratio_threshold = 0.3  # 두 번째 얼굴 면적이 최대 얼굴의 30% 이하이면 단일로 침
                is_single_dominant = (len(face_locations) > 1 and second_largest < (largest_area * ratio_threshold))

                for i, (top, right, bottom, left) in enumerate(face_locations):
                    draw.rectangle(((left, top), (right, bottom)), outline="red", width=5)
                    if i != main_index:
                        face_region = pil_image.crop((left, top, right, bottom))
                        blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=15))
                        pil_image.paste(blurred_face, (left, top, right, bottom))
                    else:
                        draw.text((left, bottom + 5), matched_person, fill="red", font=font)

            # -------------------------------------------------------------------
            # 사람 이름별 폴더 생성 및 원본 파일 복사
            #    * (수정) 여러 얼굴이 있지만 한 명이 압도적으로 크면 "단일 얼굴" 폴더에 저장
            #    * 그 외 진짜 단체사진일 경우 -> output_group
            # -------------------------------------------------------------------
            if matched_person != "unknown":
                if len(face_locations) > 1 and not is_single_dominant:
                    # 실제로 여러 얼굴
                    output_group_path = os.path.join(base_output_folder, "output_group", matched_person)
                    os.makedirs(output_group_path, exist_ok=True)
                    logger.info(f"[process_single_image] Copying original (multi-face) to {output_group_path}")
                    shutil.copy(file_path, os.path.join(output_group_path, file))
                else:
                    # 단일 얼굴 (또는 압도적으로 큰 얼굴 1명)
                    single_output_path = os.path.join(base_output_folder, matched_person)
                    os.makedirs(single_output_path, exist_ok=True)
                    logger.info(f"[process_single_image] Copying original (single-face) to {single_output_path}")
                    shutil.copy(file_path, os.path.join(single_output_path, file))
            else:
                logger.info("[process_single_image] matched_person is unknown, already handled.")

            resized = pil_image.copy()
            resized.thumbnail((600, 400), Image.Resampling.LANCZOS)

            logger.info(f"[process_single_image] Returning thumbnail for file: {file}, matched_person={matched_person}")
            return file, matched_person, resized

        except Exception as e:
            logger.exception(f"[process_single_image] Error processing {file}")
            return file, "unknown", None
        finally:
            self._log_memory_usage("Finally (process_single_image)")

    def process_images_in_background(self, dataset_folder, base_output_folder, known_images_folder):
        logger.info(f"[process_images_in_background] Called with dataset={dataset_folder}, base_output={base_output_folder}, known_images={known_images_folder}")
        self._log_memory_usage("Start of process_images_in_background")
        try:
            logger.info("[process_images_in_background] Scanning known folder...")
            known_faces = []
            known_names = []
            for known_file in os.listdir(known_images_folder):
                logger.info(f"[process_images_in_background] Checking {known_file} in known_images_folder")
                try:
                    if known_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        name, _ = os.path.splitext(known_file)
                        img_path = os.path.join(known_images_folder, known_file)
                        logger.info(f"[process_images_in_background] Loading known image {img_path} for name {name}")
                        known_img = face_recognition.load_image_file(img_path)
                        encs = face_recognition.face_encodings(known_img)
                        if encs:
                            known_names.append(name)
                            known_faces.append(encs[0])
                            logger.info(f"[process_images_in_background] Appended known face {name}")
                except Exception as e:
                    logger.exception("Error loading known file:", known_file, e)

            output_path_unknown = os.path.join(base_output_folder, "output_unknown")
            os.makedirs(output_path_unknown, exist_ok=True)
            logger.info(f"[process_images_in_background] Prepared output_unknown folder: {output_path_unknown}")

            output_group_root = os.path.join(base_output_folder, "output_group")
            os.makedirs(output_group_root, exist_ok=True)
            logger.info(f"[process_images_in_background] Created output_group root folder: {output_group_root}")

            all_files = [f for f in os.listdir(dataset_folder)
                         if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            total_files = len(all_files)
            logger.info(f"[process_images_in_background] Found {total_files} image files to process")
            done_count = 0
            person_counts = {}
            # max_workers = os.cpu_count() // 2
            max_workers = 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for f in all_files:
                    self._log_memory_usage(f"Submitting {f}")
                    ft = executor.submit(
                        self.process_single_image,
                        f, dataset_folder, known_faces, known_names,
                        base_output_folder, output_path_unknown
                    )
                    future_map[ft] = f

                for ft in concurrent.futures.as_completed(future_map):
                    file, matched_person, thumb = ft.result()
                    done_count += 1
                    logger.info(f"[process_images_in_background] Completed {file}. matched_person={matched_person}")
                    if matched_person:
                        person_counts[matched_person] = person_counts.get(matched_person, 0) + 1

                    progress_percent = done_count / total_files * 100
                    self.results_queue.put({
                        "progress_percent": progress_percent,
                        "thumbnail": thumb,
                        "person_counts": None
                    })
                    self._log_memory_usage(f"Completed {file}")

            # 모든 작업 종료 후
            logger.info("[process_images_in_background] All tasks completed. Finalizing.")
            self.results_queue.put({
                "progress_percent": 100.0,
                "thumbnail": None,
                "person_counts": person_counts
            })

        except Exception as e:
            logger.exception("[process_images_in_background] Fatal error")
            self.results_queue.put({
                "progress_percent": 100,
                "thumbnail": None,
                "person_counts": {}
            })
        finally:
            self._log_memory_usage("End of process_images_in_background") 