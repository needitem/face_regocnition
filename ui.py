import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
from PIL import ImageTk

class FaceRecognitionUI:
    def __init__(self, engine):
        self.engine = engine

        # 메인 윈도우
        self.window = tk.Tk()
        self.window.title("Face Recognition GUI")

        # 설정 변수
        self.dataset_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.known_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.45)

        # 위젯 구성
        self._build_widgets()

        # 주기적으로 queue 확인
        self._check_queue_and_update()

    def _build_widgets(self):
        # Dataset
        tk.Label(self.window, text="Dataset:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.window, textvariable=self.dataset_var, width=40).grid(row=0, column=1)
        tk.Button(self.window, text="Browse", command=self._browse_dataset).grid(row=0, column=2)

        # Known images
        tk.Label(self.window, text="Known Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.window, textvariable=self.known_var, width=40).grid(row=1, column=1)
        tk.Button(self.window, text="Browse", command=self._browse_known).grid(row=1, column=2)

        # Output base
        tk.Label(self.window, text="Output Base:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.window, textvariable=self.output_var, width=40).grid(row=2, column=1)
        tk.Button(self.window, text="Browse", command=self._browse_output).grid(row=2, column=2)

        # Progress
        self.progress_label = tk.Label(self.window, text="Progress: 0%")
        self.progress_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(self.window, length=400, mode="determinate")
        self.progress_bar.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

        # Person count text
        self.text_box = tk.Text(self.window, width=50, height=8)
        self.text_box.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

        # Current image
        self.current_image_label = tk.Label(self.window)
        self.current_image_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

        # Threshold : 클 수록 덜비슷해도 같은 사람으로 처리
        tk.Label(self.window, text="Threshold(Large: 덜비슷해도 같은 사람으로 처리):").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        tk.Scale(self.window, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.01,
                 variable=self.threshold_var).grid(row=7, column=1, padx=5, pady=5)

        # Start Button
        tk.Button(self.window, text="Start", command=self._start_processing).grid(row=8, column=0, columnspan=3, pady=10)

    def _browse_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_var.set(folder)

    def _browse_known(self):
        folder = filedialog.askdirectory(title="Select Known Images Folder")
        if folder:
            self.known_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Base Folder")
        if folder:
            self.output_var.set(folder)

    def _start_processing(self):
        dataset = self.dataset_var.get()
        known_dir = self.known_var.get()
        output_base = self.output_var.get()
        if not (dataset and known_dir and output_base):
            return

        # Threshold 설정
        self.engine.set_threshold(self.threshold_var.get())

        # 별도 스레드에서 처리
        t = threading.Thread(
            target=self.engine.process_images_in_background,
            args=(dataset, output_base, known_dir),
            daemon=True
        )
        t.start()

    def _check_queue_and_update(self):
        try:
            while True:
                result = self.engine.results_queue.get_nowait()

                # 진행도
                p = result.get("progress_percent", 0)
                self.progress_label.config(text=f"Progress: {p:.2f}%")
                self.progress_bar["value"] = p

                # 썸네일
                thumb = result.get("thumbnail")
                if thumb:
                    preview = ImageTk.PhotoImage(thumb)
                    self.current_image_label.config(image=preview)
                    self.current_image_label.image = preview

                # 최종 person_counts가 있으면 text_box 업데이트
                persons = result.get("person_counts")
                if persons is not None:
                    self.text_box.delete("1.0", tk.END)
                    for name, count in persons.items():
                        self.text_box.insert(tk.END, f"{name}: {count}\n")

        except queue.Empty:
            pass

        # 0.1초 후 다시 확인
        self.window.after(100, self._check_queue_and_update)

    def run(self):
        self.window.mainloop() 