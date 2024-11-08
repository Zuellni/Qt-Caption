import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Model:
    def __init__(self, config: dict):
        self.path = config.get("model", "microsoft/Florence-2-large-ft")
        self.device = config.get("device", "cpu")
        self.dtype = "float32" if self.device == "cpu" else "float16"

        self.task = config.get("task", "<MORE_DETAILED_CAPTION>")
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.num_beams = config.get("num_beams", 3)
        self.repetition_penalty = config.get("repetition_penalty", 1.0)

        self.model = None
        self.processor = None

    def __call__(self, path: Path) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        image = Image.open(path).convert("RGB")
        dtype = getattr(torch, self.dtype)

        if not self.model:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.path,
                device_map=self.device,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval()

        if not self.processor:
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=self.path,
                device_map=self.device,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        with torch.inference_mode():
            input = self.processor(text=self.task, images=image)
            input.to(device=self.device, dtype=dtype)

            output_ids = self.model.generate(
                input_ids=input["input_ids"],
                pixel_values=input["pixel_values"],
                do_sample=False,
                early_stopping=False,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
            )

            output = self.processor.decode(
                token_ids=output_ids[0],
                skip_special_tokens=True,
            )

        output = "\n".join([t for t in output.splitlines() if t])
        output = " ".join(output.split())
        return output


class Worker(QThread):
    finished = Signal(str)

    def __init__(self, model: Model, path: Path):
        super().__init__()
        self.model = model
        self.path = path

    def run(self):
        output = self.model(self.path)
        self.finished.emit(output)


class Window(QWidget):
    def __init__(self, model: Model, config: dict, icon: str):
        super().__init__()

        self.model = model
        self.extensions = config.get(
            "extensions", (".bmp", ".jpeg", ".jpg", ".png", ".webp")
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.loading)

        self.cptn_index = 0
        self.cptn_icons = (
            "hourglass-start",
            "hourglass-half",
            "hourglass-end",
            "hourglass",
        )

        self.worker = None
        self.folder = None
        self.files = None
        self.file = None

        self.margin = 8
        self.index = 0

        self.setMinimumSize(640, 480)
        self.setWindowIcon(QIcon(icon))
        self.setWindowTitle("Qt Caption")

        self.title = QLabel("(0/0)")
        self.image = QLabel()
        self.text = QPlainTextEdit()
        self.text.setEnabled(False)

        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prev_button = QPushButton("left")
        self.open_button = QPushButton("folder")
        self.cptn_button = QPushButton("eye")
        self.save_button = QPushButton("save")
        self.next_button = QPushButton("right")

        self.prev_button.clicked.connect(self.prev_image)
        self.open_button.clicked.connect(self.open_folder)
        self.cptn_button.clicked.connect(self.caption_image)
        self.save_button.clicked.connect(self.save_caption)
        self.next_button.clicked.connect(self.next_image)

        pointer = QCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_button.setCursor(pointer)
        self.open_button.setCursor(pointer)
        self.cptn_button.setCursor(pointer)
        self.save_button.setCursor(pointer)
        self.next_button.setCursor(pointer)

        self.prev_key = config.get("previous", "Ctrl+Shift+A")
        self.prev_key = QShortcut(QKeySequence(self.prev_key), self)
        self.prev_key.activated.connect(self.prev_image)

        self.cptn_key = config.get("caption", "Ctrl+Shift+S")
        self.cptn_key = QShortcut(QKeySequence(self.cptn_key), self)
        self.cptn_key.activated.connect(self.caption_image)

        self.next_key = config.get("next", "Ctrl+Shift+D")
        self.next_key = QShortcut(QKeySequence(self.next_key), self)
        self.next_key.activated.connect(self.next_image)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(0)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.addWidget(self.prev_button, 1)
        self.button_layout.addWidget(self.open_button, 1)
        self.button_layout.addWidget(self.cptn_button, 1)
        self.button_layout.addWidget(self.save_button, 1)
        self.button_layout.addWidget(self.next_button, 1)

        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(*[self.margin // 2] * 4)
        self.layout.addWidget(self.title, 0, 0, 1, 2)
        self.layout.addWidget(self.image, 1, 0, 1, 1)
        self.layout.addWidget(self.text, 1, 1, 1, 1)
        self.layout.addLayout(self.button_layout, 2, 0, 1, 2)

        self.setLayout(self.layout)
        self.show()

    def open_folder(self):
        self.folder = QFileDialog.getExistingDirectory(self)

        if self.folder:
            self.files = [
                file
                for file in Path(self.folder).glob("*.*")
                if file.suffix in self.extensions
            ]

            self.index = 0
            self.show_image()

    def caption_image(self):
        if self.file:
            self.prev_button.setEnabled(False)
            self.open_button.setEnabled(False)
            self.cptn_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.next_button.setEnabled(False)

            self.cptn_index = 0
            self.cptn_button.setText("hourglass")
            self.timer.start(500)

            self.worker = Worker(self.model, self.file)
            self.worker.finished.connect(self.finished)
            self.worker.start()

    def loading(self):
        self.cptn_button.setText(self.cptn_icons[self.cptn_index])
        self.cptn_index = (self.cptn_index + 1) % len(self.cptn_icons)

    def finished(self, output: str):
        self.prev_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.cptn_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.next_button.setEnabled(True)

        self.timer.stop()
        self.cptn_button.setText("eye")
        self.text.setPlainText(output)
        self.save_caption()

    def save_caption(self):
        if self.file:
            text = self.text.toPlainText().strip()
            path = self.file.parent / f"{self.file.stem}.txt"
            path.write_text(text, encoding="utf-8") if text else path.unlink(True)

    def prev_image(self):
        if self.files:
            self.index = (self.index - 1) % len(self.files)
            self.show_image()

    def next_image(self):
        if self.files:
            self.index = (self.index + 1) % len(self.files)
            self.show_image()

    def show_image(self):
        if self.files:
            self.save_caption()
            self.file = self.files[self.index]
            self.title.setText(f"{self.file.name} ({self.index + 1}/{len(self.files)})")

            pixmap = QPixmap(self.file.as_posix())
            scaled = pixmap.scaled(
                self.text.width() - self.margin,
                self.text.height() - self.margin,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            rounded = QPixmap(scaled.size())
            rounded.fill(Qt.GlobalColor.transparent)

            painter = QPainter(rounded)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
            )
            painter.setBrush(QBrush(scaled))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(scaled.rect(), self.margin, self.margin)
            painter.end()

            self.image.setPixmap(rounded)
            self.image.setStyleSheet("background: 0; padding: 0")

            path = self.file.parent / f"{self.file.stem}.txt"
            text = path.read_text(encoding="utf-8") if path.exists() else ""

            self.text.setPlainText(text)
            self.text.setEnabled(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.width() // 2 - self.margin // 2
        self.image.setFixedWidth(width)
        self.text.setFixedWidth(width)
        self.show_image()

    def closeEvent(self, event):
        self.save_caption()
        super().closeEvent(event)


if __name__ == "__main__":
    dir = Path(__file__).parent
    assets = dir / "assets"

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=dir / "config.json")
    args = parser.parse_args()

    config = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if args.config.is_file()
        else {}
    )

    style = (assets / "style.qss").read_text(encoding="utf-8")
    icon = (assets / "icon.png").as_posix()

    model = Model(config)
    app = QApplication([])
    window = Window(model, config, icon)

    QFontDatabase.addApplicationFont((assets / "main.ttf").as_posix())
    QFontDatabase.addApplicationFont((assets / "icon.otf").as_posix())

    app.setStyleSheet(style)
    sys.exit(app.exec())
