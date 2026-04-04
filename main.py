import sys
import os
import cv2
import numpy as np
import multiprocessing
import tempfile
from PySide6.QtWidgets import (QLabel, QComboBox, QPushButton, QFrame, QLineEdit, 
                             QStackedWidget, QFileDialog, QApplication, QMainWindow, 
                             QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QMessageBox, 
                             QGraphicsDropShadowEffect, QSlider, QScrollArea, QSpinBox,
                             QListWidget, QListWidgetItem, QListView)
from PySide6.QtCore import (QPoint, QRect, QPointF, QRectF, Qt, Signal, Slot, QUrl, 
                          QMimeData, QThread, QPropertyAnimation, QTimer, QElapsedTimer, QSize)
import uuid
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QPen, QColor, QRegion, QDrag, QFont, QIcon

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    ort = None

# Local model paths
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUPERRES_PATH = os.path.join(BASE_DIR, "4k", "4x-ClearRealityV1.onnx")

# ══════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════
class ImageProcessor:
    def crop(self, img, x1, y1, x2, y2):
        if img is None: return None
        try:
            h, w = img.shape[:2]
            x1, x2 = sorted([max(0, min(w, x1)), max(0, min(w, x2))])
            y1, y2 = sorted([max(0, min(h, y1)), max(0, min(h, y2))])
            if x2-x1 < 1 or y2-y1 < 1: return None
            return img[y1:y2, x1:x2].copy()
        except: return None
    
    def resize(self, img, w, h):
        if img is None: return None
        return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)
    
    def adjust_brightness(self, img, value):
        if img is None: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] + value, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, img, value):
        if img is None: return None
        factor = (value + 100) / 100.0
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    
    def adjust_saturation(self, img, value):
        if img is None: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (value / 100.0), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def rotate(self, img, angle):
        if img is None: return None
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    def blur(self, img, value):
        if img is None: return None
        ksize = max(1, int(value))
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    def apply_super_res(self, img):
        if img is None or ort is None or not os.path.exists(SUPERRES_PATH): return None
        try:
            so = ort.SessionOptions(); threads = min(4, max(1, multiprocessing.cpu_count() // 2))
            so.intra_op_num_threads = threads; so.inter_op_num_threads = threads
            sess = ort.InferenceSession(SUPERRES_PATH, sess_options=so, providers=["CPUExecutionProvider"])
            inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), 0)
            out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
            out = np.clip(out[0] * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(np.transpose(out, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        except Exception as e: print(f"AI error: {e}"); return None

class Worker(QThread):
    finished = Signal(object)
    def __init__(self, func): super().__init__(); self.func = func
    def run(self):
        res = self.func()
        self.finished.emit(res if isinstance(res, np.ndarray) else None)

# ══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setAttribute(Qt.WA_TransparentForMouseEvents, False); self._dots = [0.2] * 7
        self._colors = [QColor("#ffff00"), QColor("#76ff03"), QColor("#f06292"), QColor("#4fc3f7"), QColor("#ba68c8"), QColor("#f57c00"), QColor("#673ab7")]
        self._timer = QTimer(self); self._timer.timeout.connect(self._update_anim); self._start_time = QElapsedTimer(); self.hide()
    def showEvent(self, e): 
        if self.parent(): self.resize(self.parent().size())
        self._start_time.start(); self._timer.start(32); super().showEvent(e)
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.parent(): self.resize(self.parent().size())
    def _update_anim(self):
        el = self._start_time.elapsed() / 1000.0
        for i in range(7):
            t = (el + (i * 0.2)) % 1.6
            self._dots[i] = 0.2 + 0.8 * (t / 0.32) if t < 0.32 else (1.0 - 0.8 * ((t - 0.32) / 0.32) if t < 0.64 else 0.2)
        self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(0, 0, 0, 220)); p.setRenderHint(QPainter.Antialiasing)
        sp = 30; sx = (self.width() - (sp * 6)) / 2; cy = self.height() / 2
        for i in range(7):
            s = self._dots[i]; r = 10 * s; rect = QRectF(sx + i * sp - r, cy - r, r*2, r*2)
            p.setBrush(self._colors[i]); p.setPen(Qt.NoPen); p.drawEllipse(rect)

class GradientButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(46)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f79533, stop:0.14 #f37055, stop:0.28 #ef4e7b,
                    stop:0.42 #a166ab, stop:0.56 #5073b8, stop:0.7 #1098ad,
                    stop:0.84 #07b39b, stop:1 #6fba82);
                color: transparent;
                border: none;
                border-radius: 23px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f79533, stop:0.14 #f37055, stop:0.28 #ef4e7b,
                    stop:0.42 #a166ab, stop:0.56 #5073b8, stop:0.7 #1098ad,
                    stop:0.84 #07b39b, stop:1 #6fba82);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f79533, stop:0.14 #f37055, stop:0.28 #ef4e7b,
                    stop:0.42 #a166ab, stop:0.56 #5073b8, stop:0.7 #1098ad,
                    stop:0.84 #07b39b, stop:1 #6fba82);
            }
        """)
        
        # Create shadow effect
        self.glow = QGraphicsDropShadowEffect(self)
        self.glow.setBlurRadius(25)
        self.glow.setColor(QColor("#a166ab"))
        self.glow.setOffset(0, 0)
        self.glow.setEnabled(False)
        self.setGraphicsEffect(self.glow)
        
    def paintEvent(self, e):
        # Draw gradient border first
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the gradient background
        rect = self.rect()
        painter.setPen(Qt.NoPen)
        
        # Create gradient
        from PySide6.QtGui import QLinearGradient
        gradient = QLinearGradient(0, 0, rect.width(), rect.height())
        gradient.setColorAt(0, QColor("#f79533"))
        gradient.setColorAt(0.14, QColor("#f37055"))
        gradient.setColorAt(0.28, QColor("#ef4e7b"))
        gradient.setColorAt(0.42, QColor("#a166ab"))
        gradient.setColorAt(0.56, QColor("#5073b8"))
        gradient.setColorAt(0.7, QColor("#1098ad"))
        gradient.setColorAt(0.84, QColor("#07b39b"))
        gradient.setColorAt(1, QColor("#6fba82"))
        
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(rect, 23, 23)
        
        # Draw inner dark background
        inner_rect = rect.adjusted(2, 2, -2, -2)
        painter.setBrush(QColor(19, 20, 22))
        painter.drawRoundedRect(inner_rect, 21, 21)
        
        # Draw text
        painter.setPen(QColor(235, 235, 235))
        font = QFont()
        font.setBold(True)
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self.text())
        
    def enterEvent(self, e):
        self.glow.setEnabled(True)
        super().enterEvent(e)
        
    def leaveEvent(self, e):
        self.glow.setEnabled(False)
        super().leaveEvent(e)

class TitleBar(QWidget):
    def __init__(self, parent, title="EDIT IMG TOOL"):
        super().__init__(parent)
        self.parent_win = parent
        self._drag_pos = None
        self.setFixedHeight(34)
        self.setStyleSheet("background:#2a2a2a; border-bottom:1px solid #1b1b1b;")
        
        l = QHBoxLayout(self)
        l.setContentsMargins(15, 0, 8, 0)
        
        t = QLabel(title)
        t.setStyleSheet("color:#e0e0e0; font-size:8px; font-weight:bold; letter-spacing:2px; background:transparent;")
        l.addWidget(t)
        l.addStretch()
        
        if "INDEPENDENT" in title:
            # Minimize button
            min_btn = QPushButton("─")
            min_btn.setFixedSize(28, 24)
            min_btn.clicked.connect(parent.showMinimized)
            min_btn.setStyleSheet("QPushButton { background:#444; color:#fff; border:none; border-radius:3px; font-size:14px; }")
            l.addWidget(min_btn)
            
            # Maximize/Restore button
            max_btn = QPushButton("□")
            max_btn.setFixedSize(28, 24)
            max_btn.clicked.connect(self.toggle_maximize)
            max_btn.setStyleSheet("QPushButton { background:#444; color:#fff; border:none; border-radius:3px; font-size:12px; }")
            l.addWidget(max_btn)
            self.max_btn = max_btn
            
            # Close button
            close_btn = QPushButton("✕")
            close_btn.setFixedSize(28, 24)
            close_btn.clicked.connect(parent.close)
            close_btn.setStyleSheet("QPushButton { background:#ff5f40; color:#fff; border:none; border-radius:3px; font-size:12px; }")
            l.addWidget(close_btn)
    
    def toggle_maximize(self):
        if self.parent_win.isMaximized():
            self.parent_win.showNormal()
            self.max_btn.setText("□")
        else:
            self.parent_win.showMaximized()
            self.max_btn.setText("❐")
            
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: 
            self._drag_pos = e.globalPosition().toPoint()
            
    def mouseMoveEvent(self, e):
        if self._drag_pos and not self.parent_win.isMaximized(): 
            self.parent_win.move(self.parent_win.pos() + e.globalPosition().toPoint() - self._drag_pos)
            self._drag_pos = e.globalPosition().toPoint()
            
    def mouseReleaseEvent(self, e): 
        self._drag_pos = None
    
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.toggle_maximize()

# Unified Drop & Drag Export Logic
class DraggableImageLabel(QLabel):
    imageDropped = Signal(object)
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self._pix = None
        self.setStyleSheet("background:#080808; color:#444; border:1px solid #2a2a2a; border-radius:4px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    def set_pix(self, pix):
        self._pix = pix
        if pix.isNull():
            self.setText(self.text())
            return
        # If pixmap is larger than necessary, use a faster initial scale or just let paintEvent handle it
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setStyleSheet("background:transparent; border:none;")

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage() or e.mimeData().hasUrls(): 
            e.accept()
            self.setStyleSheet("border:2px dashed #ba68c8; background:#1a101a;")
            
    def dragLeaveEvent(self, e):
        self.setStyleSheet("background:#080808; border:1px solid #ba68c8; border-radius:4px;" if self._pix else "background:#080808; color:#444; border:1px solid #2a2a2a; border-radius:4px;")
        
    def dropEvent(self, e):
        if e.mimeData().hasUrls(): 
            paths = []
            for u in e.mimeData().urls():
                if u.isLocalFile(): paths.append(u.toLocalFile())
                elif u.scheme() in ('http', 'https'): paths.append(u.toString())
            if not paths: return
            if len(paths) == 1 and not (not paths[0].startswith('http') and os.path.exists(paths[0]) and os.path.isdir(paths[0])):
                self.imageDropped.emit(paths[0])
            else:
                self.imageDropped.emit(paths)
        elif e.mimeData().hasImage(): 
            img = e.mimeData().imageData()
            if isinstance(img, QImage):
                img = img.convertToFormat(QImage.Format_RGB32)
                arr = np.array(img.bits()).reshape((img.height(), img.width(), 4))
                self.imageDropped.emit(cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._pix:
            drag = QDrag(self)
            md = QMimeData()
            img = self._pix.toImage()
            md.setImageData(img)
            # Create temp file for Desktop Export
            tmp = os.path.join(tempfile.gettempdir(), f"drag_export_{uuid.uuid4().hex[:8]}.png")
            img.save(tmp, "PNG")
            md.setUrls([QUrl.fromLocalFile(tmp)])
            drag.setMimeData(md)
            drag.setPixmap(self._pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            drag.exec(Qt.CopyAction)

# ══════════════════════════════════════════════════════════════
# CANVAS
# ══════════════════════════════════════════════════════════════
class ImageCanvas(QWidget):
    cropChanged = Signal(object)
    imageDropped = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self._pix = None
        self.cv_img = None
        self.crop_mode = False
        self.aspect_ratio = None
        self._crop_s = self._crop_e = None
        self._active_h = None
        self._crop_m = False
        self.scale_f = 1.0
        self.offset = QPointF(0,0)
        self.space_h = False
        self._last_d = None

    def set_image(self, img):
        self.cv_img = img
        h, w = img.shape[:2]
        qi = QImage(img.tobytes(), w, h, 3*w, QImage.Format_BGR888)
        self._pix = QPixmap.fromImage(qi)
        self.scale_f = 1.0
        self.offset = QPointF(0,0)
        self._crop_s = self._crop_e = None
        self.update()

    def _img_rect(self):
        if not self._pix: return QRectF()
        s = self._pix.size()
        s.scale(self.size(), Qt.KeepAspectRatio)
        s = s * self.scale_f
        return QRectF((self.width()-s.width())/2 + self.offset.x(), 
                     (self.height()-s.height())/2 + self.offset.y(), 
                     s.width(), s.height())

    def widget_to_image(self, pt):
        r = self._img_rect()
        if r.width() <= 0 or r.height() <= 0 or self.cv_img is None: 
            return QPoint(0,0)
        h, w = self.cv_img.shape[:2]
        return QPoint(int((pt.x()-r.x())/r.width()*w), 
                     int((pt.y()-r.y())/r.height()*h))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._pix:
            r = self._img_rect()
            p.drawPixmap(r.toRect(), self._pix)
            if self.crop_mode and self._crop_s and self._crop_e:
                rect = QRect(self._crop_s, self._crop_e).normalized()
                p.setPen(QPen(QColor("#4fc3f7"), 2))
                p.drawRect(rect)
                for h in self._get_handles().values(): 
                    p.setBrush(QColor("#4fc3f7"))
                    p.drawEllipse(h, 5, 5)

    def _get_handles(self):
        if not self._crop_s or not self._crop_e: return {}
        r = QRect(self._crop_s, self._crop_e).normalized()
        l, re, t, b = r.left(), r.right(), r.top(), r.bottom()
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": QPoint(l, t), "tr": QPoint(re, t), 
            "bl": QPoint(l, b), "br": QPoint(re, b), 
            "t": QPoint(cx, t), "b": QPoint(cx, b), 
            "l": QPoint(l, cy), "r": QPoint(re, cy)
        }

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
            
    def dropEvent(self, e):
        p = e.mimeData().urls()[0].toLocalFile()
        with open(p, 'rb') as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None: self.imageDropped.emit(img)

    def mousePressEvent(self, e):
        p = e.position().toPoint()
        if self.space_h: 
            self._last_d = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
            
        if self.crop_mode:
            for n, pt in self._get_handles().items():
                if (p - pt).manhattanLength() < 15: 
                    self._active_h = n
                    return
                    
            if self._crop_s and self._crop_e and QRect(self._crop_s, self._crop_e).normalized().contains(p):
                self._crop_m = True
                self._os = p-self._crop_s
                self._oe = p-self._crop_e
                return
                
            self._crop_s = p
            self._crop_e = p
            
            # Apply aspect ratio immediately when starting new crop
            if self.aspect_ratio:
                self._crop_e = QPoint(p.x() + 100, p.y() + int(100 / self.aspect_ratio))
                
            self.update()
        elif self._pix:
            drag = QDrag(self)
            md = QMimeData()
            qi = self._pix.toImage()
            md.setImageData(qi)
            tmp = os.path.join(tempfile.gettempdir(), f"canvas_export_{uuid.uuid4().hex[:8]}.png")
            qi.save(tmp, "PNG")
            md.setUrls([QUrl.fromLocalFile(tmp)])
            drag.setMimeData(md)
            drag.setPixmap(self._pix.scaled(80,80))
            drag.exec(Qt.CopyAction)

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        if self.space_h and self._last_d:
            delta = e.position() - self._last_d
            self.offset += delta
            self._last_d = e.position()
            self.update()
            return
            
        if self.crop_mode and e.buttons() & Qt.LeftButton:
            if self._active_h:
                r = QRect(self._crop_s, self._crop_e).normalized()
                l, re, t, b = r.left(), r.right(), r.top(), r.bottom()
                h = self._active_h
                
                if "t" in h: t = p.y()
                if "b" in h: b = p.y()
                if "l" in h: l = p.x()
                if "r" in h: re = p.x()
                
                if self.aspect_ratio:
                    nw, nh = abs(re-l), abs(b-t)
                    if nw>0 and nh>0:
                        if h in ("l", "r"): 
                            nh = int(nw/self.aspect_ratio)
                        elif h in ("t", "b"): 
                            nw = int(nh*self.aspect_ratio)
                        else:
                            if nw/nh > self.aspect_ratio: 
                                nw = int(nh*self.aspect_ratio)
                            else: 
                                nh = int(nw/self.aspect_ratio)
                                
                        if "l" in h: l = re-nw if re>l else re+nw
                        if "r" in h: re = l+nw if re>l else l-nw
                        if "t" in h: t = b-nh if b>t else b+nh
                        if "b" in h: b = t+nh if b>t else t-nh
                        
                self._crop_s = QPoint(l, t)
                self._crop_e = QPoint(re, b)
            elif self._crop_m: 
                self._crop_s = p - self._os
                self._crop_e = p - self._oe
            else:
                self._crop_e = p
                if self.aspect_ratio:
                    nw = abs(p.x()-self._crop_s.x())
                    nh = int(nw/self.aspect_ratio)
                    self._crop_e = QPoint(
                        self._crop_s.x()+(nw if p.x()>self._crop_s.x() else -nw), 
                        self._crop_s.y()+(nh if p.y()>self._crop_s.y() else -nh)
                    )
            self._update_preview()
            self.update()

    def mouseReleaseEvent(self, e):
        self._active_h = None
        self._crop_m = False
        self._last_d = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _update_preview(self):
        if not self._crop_s or not self._crop_e: return
        r = QRect(self._crop_s, self._crop_e).normalized()
        p1, p2 = self.widget_to_image(r.topLeft()), self.widget_to_image(r.bottomRight())
        crop = ImageProcessor().crop(self.cv_img, p1.x(), p1.y(), p2.x(), p2.y())
        if crop is not None:
            h, w = crop.shape[:2]
            qi = QImage(crop.tobytes(), w, h, 3*w, QImage.Format_BGR888)
            self.cropChanged.emit(QPixmap.fromImage(qi))

# ═════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═════════════════════════════════════════════════════════════
class CropToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1200, 700)
        self.setStyleSheet("QMainWindow { background:#1b1b1b; }")
        self.history = []
        self.redo_s = []
        self.proc = ImageProcessor()
        self.current_img = None
        self.batch_running = False
        self._build_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def stop_batch(self):
        self.batch_running = False
        try: self.loading.hide()
        except: pass
        for attr in ['loader', 'loadw', 'rload', 'worker']:
            w = getattr(self, attr, None)
            if w and w.isRunning():
                try: w.terminate(); w.wait()
                except: pass
        if hasattr(self, 'btn_stop_batch'): self.btn_stop_batch.hide()
        
    def closeEvent(self, e):
        self.stop_batch()
        super().closeEvent(e)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)
        
        # SIDEBAR
        sb = QFrame()
        sb.setFixedWidth(140)
        sb.setStyleSheet("background:#151515; border-right:1px solid #2a2a2a;")
        lay.addWidget(sb)
        
        sl = QVBoxLayout(sb)
        sl.setContentsMargins(10, 20, 10, 20)
        sl.setSpacing(10)
        
        logo = QLabel("IMG TOOL")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("color:#e0e0e0; font-size:12px; font-weight:bold; letter-spacing:2px; margin-bottom:15px;")
        sl.addWidget(logo)
        
        self.btn_open = GradientButton("OPEN")
        self.btn_open.clicked.connect(self.browse)
        sl.addWidget(self.btn_open)
        
        self.btn_new = GradientButton("NEW")
        self.btn_new.clicked.connect(self.reset_workspace)
        sl.addWidget(self.btn_new)
        
        self.btn_save = GradientButton("SAVE")
        self.btn_save.clicked.connect(self.save_edit)
        sl.addWidget(self.btn_save)
        
        sl.addSpacing(10)
        
        self.btn_v = GradientButton("DRAG (V)")
        self.btn_v.setCheckable(True)
        self.btn_v.setChecked(True)
        self.btn_v.clicked.connect(self.set_v)
        sl.addWidget(self.btn_v)
        
        self.btn_c = GradientButton("CROP (C)")
        self.btn_c.setCheckable(True)
        self.btn_c.clicked.connect(self.set_c)
        sl.addWidget(self.btn_c)
        
        sl.addSpacing(10)
        
        self.btn_undo = GradientButton("UNDO")
        self.btn_undo.clicked.connect(self.undo)
        sl.addWidget(self.btn_undo)
        
        self.btn_redo = GradientButton("REDO")
        self.btn_redo.clicked.connect(self.redo)
        sl.addWidget(self.btn_redo)
        
        sl.addSpacing(15)
        
        # Image adjustment controls
        adj_title = QLabel("ADJUSTMENTS")
        adj_title.setAlignment(Qt.AlignCenter)
        adj_title.setStyleSheet("color:#e0e0e0; font-size:10px; font-weight:bold; margin-bottom:5px;")
        sl.addWidget(adj_title)
        
        def create_adj_row(label_text, min_v, max_v, default_v, callback):
            row = QFrame()
            row.setStyleSheet("background:transparent; border:none;")
            rlay = QVBoxLayout(row)
            rlay.setContentsMargins(0,2,0,5)
            rlay.setSpacing(2)
            
            hlay = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color:#aaa; font-size:9px; font-weight:bold;")
            hlay.addWidget(lbl)
            
            val_lbl = QLabel(str(default_v))
            val_lbl.setStyleSheet("color:#888; font-size:8px;")
            hlay.addStretch()
            hlay.addWidget(val_lbl)
            rlay.addLayout(hlay)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default_v)
            slider.setFixedHeight(14)
            slider.setStyleSheet("""
                QSlider::groove:horizontal { background: #222; height: 3px; border-radius: 1px; }
                QSlider::handle:horizontal { background: #ba68c8; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
            """)
            
            def on_val(v):
                val_lbl.setText(str(v))
                callback(v)
            
            slider.valueChanged.connect(on_val)
            rlay.addWidget(slider)
            return row, slider

        self.adj_rows = {}
        
        # Brightness
        r_br, self.sb_br = create_adj_row("BRT", -100, 100, 0, self.sidebar_adj_br)
        sl.addWidget(r_br)
        
        # Contrast
        r_ct, self.sb_ct = create_adj_row("CON", -50, 50, 0, self.sidebar_adj_ct)
        sl.addWidget(r_ct)
        
        # Saturation
        r_st, self.sb_st = create_adj_row("SAT", 0, 200, 100, self.sidebar_adj_st)
        sl.addWidget(r_st)
        
        # Rotation
        r_rt, self.sb_rt = create_adj_row("ROT", -180, 180, 0, self.sidebar_adj_rt)
        sl.addWidget(r_rt)
        
        # Blur
        r_bl, self.sb_bl = create_adj_row("BLR", 0, 50, 0, self.sidebar_adj_bl)
        sl.addWidget(r_bl)
        
        sl.addSpacing(10)
        
        self.btn_commit_adj = GradientButton("COMMIT ADJ")
        self.btn_commit_adj.setFixedHeight(30)
        self.btn_commit_adj.clicked.connect(self.commit_sidebar_adj)
        sl.addWidget(self.btn_commit_adj)
        
        self.btn_reset_adj = QPushButton("RESET")
        self.btn_reset_adj.setStyleSheet("color:#888; font-size:10px; background:transparent; border:none;")
        self.btn_reset_adj.clicked.connect(self.reset_sidebar_adj)
        sl.addWidget(self.btn_reset_adj)
        
        sl.addStretch()
        
        eb = QPushButton("EXIT")
        eb.clicked.connect(self.close)
        eb.setStyleSheet("color:#ff5f40; font-weight:bold; background:transparent; border:none;")
        sl.addWidget(eb)
        
        # CENTER (Editor - STRETCH 2)
        cnt = QFrame()
        cnt.setStyleSheet("background:#1b1b1b;")
        lay.addWidget(cnt, 2)
        
        cl = QVBoxLayout(cnt)
        cl.setContentsMargins(0,0,0,0)
        cl.setSpacing(0)
        
        cl.addWidget(TitleBar(self, "Yasser-27 : Github"))
        
        sub = QFrame()
        sub.setFixedHeight(40)
        sub.setStyleSheet("background:#1b1b1b; border-bottom:1px solid #2a2a2a;")
        cl.addWidget(sub)
        
        sh = QHBoxLayout(sub)
        sh.setContentsMargins(10, 0, 10, 0)
        
        ratio_label = QLabel("RATIO:")
        ratio_label.setStyleSheet("color:#e0e0e0; font-size:11px;")
        sh.addWidget(ratio_label)
        
        self.ratio = QComboBox()
        self.ratio.addItems(["Free", "1:1", "16:9", "4:3"])
        self.ratio.currentIndexChanged.connect(self.on_ratio)
        self.ratio.setStyleSheet("""
            QComboBox { 
                background:#151515; 
                color:#e0e0e0; 
                border: 1px solid #2a2a2a;
                padding: 5px;
                border-radius: 3px;
            } 
            QComboBox QAbstractItemView { 
                background:#151515; 
                color:#e0e0e0; 
                selection-background-color:#444; 
                selection-color:#ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #e0e0e0;
                margin-right: 5px;
            }
        """)
        sh.addWidget(self.ratio)
        
        sh.addStretch()
        
        w_label = QLabel("W:")
        w_label.setStyleSheet("color:#e0e0e0; font-size:11px;")
        sh.addWidget(w_label)
        
        self.inp_w = QLineEdit("0")
        self.inp_w.setFixedWidth(60)
        self.inp_w.setStyleSheet("""
            QLineEdit {
                background:#151515; 
                color:#e0e0e0; 
                border: 1px solid #2a2a2a;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        sh.addWidget(self.inp_w)
        
        x_label = QLabel("x")
        x_label.setStyleSheet("color:#e0e0e0; font-size:11px;")
        sh.addWidget(x_label)
        
        self.inp_h = QLineEdit("0")
        self.inp_h.setFixedWidth(60)
        self.inp_h.setStyleSheet("""
            QLineEdit {
                background:#151515; 
                color:#e0e0e0; 
                border: 1px solid #2a2a2a;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        sh.addWidget(self.inp_h)
        
        ba = GradientButton("Apply")
        ba.setFixedWidth(80)
        ba.setFixedHeight(32)
        ba.clicked.connect(self.apply_res)
        sh.addWidget(ba)
        
        self.stack = QStackedWidget()
        cl.addWidget(self.stack, 1)
        self.loading_crop = LoadingOverlay(self.stack)
        
        self.cdrop = QFrame()
        cdl = QVBoxLayout(self.cdrop)
        cdl.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.cdrop)
        
        t_cd = QLabel("DRAG IMAGE TO EDIT")
        t_cd.setStyleSheet("color:#444; font-weight:bold; font-size:16px;")
        cdl.addWidget(t_cd)
        
        self.cdrop.setAcceptDrops(True)
        self.cdrop.dragEnterEvent = lambda e: e.accept()
        def _safe_cdrop(e):
            if e.mimeData().hasUrls() and e.mimeData().urls():
                u = e.mimeData().urls()[0]
                self.handle_files(u.toLocalFile() if u.isLocalFile() else u.toString())
            elif e.mimeData().hasImage():
                img = e.mimeData().imageData()
                if isinstance(img, QImage):
                    img = img.convertToFormat(QImage.Format_RGB32)
                    arr = np.array(img.bits()).reshape((img.height(), img.width(), 4))
                    self.handle_files(cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR))
        self.cdrop.dropEvent = _safe_cdrop
        
        cedit = QWidget()
        cel = QHBoxLayout(cedit)
        cel.setContentsMargins(15,15,15,15)
        cel.setSpacing(20)
        
        # CROP EDIT (Canvas)
        self.canvas = ImageCanvas()
        self.canvas.cropChanged.connect(self.upd_crop_prev)
        self.canvas.imageDropped.connect(self.handle_files)
        self.canvas.setStyleSheet("background:#151515; border:1px solid #2a2a2a; border-radius:4px;")
        cel.addWidget(self.canvas, 3)
        
        # CROP RESULT (Preview)
        self.crop_prev = DraggableImageLabel("REVIEW CROP EXPORT")
        self.crop_prev.setFixedWidth(300)
        self.crop_prev.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        cel.addWidget(self.crop_prev, 1)
        
        self.stack.addWidget(cedit)
        
        # RIGHT (4K AI - FIXED WIDTH)
        right = QFrame()
        right.setFixedWidth(320)
        right.setStyleSheet("background:#151515; border-left:1px solid #2a2a2a;")
        lay.addWidget(right)
        
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0,0,0,0)
        rl.setSpacing(0)
        rl.addWidget(TitleBar(self, "4K INDEPENDENT"))
        
        rc = QWidget()
        rcl = QVBoxLayout(rc)
        rcl.setContentsMargins(20, 20, 20, 20)
        rcl.setSpacing(15)
        rl.addWidget(rc)
        
        self.r_src = DraggableImageLabel("4K SOURCE")
        self.r_src.setFixedHeight(160)
        self.r_src.imageDropped.connect(self.handle_rdrop)
        rcl.addWidget(self.r_src)
        
        self.r_res = DraggableImageLabel("4K RESULT")
        self.r_res.setFixedHeight(160)
        rcl.addWidget(self.r_res)
        
        rbl = QHBoxLayout()
        rbl.setSpacing(10)
        rcl.addLayout(rbl)
        
        self.btn_sr = GradientButton("4K SOURCE")
        self.btn_sr.setFixedHeight(36)
        self.btn_sr.clicked.connect(lambda: self.run_4k(self._last_rsrc))
        rbl.addWidget(self.btn_sr)
        
        self.btn_save_sr = GradientButton("SAVE 4K")
        self.btn_save_sr.setFixedHeight(36)
        self.btn_save_sr.clicked.connect(self.save_sr)
        rbl.addWidget(self.btn_save_sr)
        
        rcl.addStretch()
        
        self.batch_grid = QListWidget()
        self.batch_grid.setViewMode(QListView.IconMode)
        self.batch_grid.setIconSize(QSize(64, 64))
        self.batch_grid.setResizeMode(QListView.Adjust)
        self.batch_grid.setSpacing(5)
        self.batch_grid.setStyleSheet("""
            QListWidget {
                background: #080808;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                color: #e0e0e0;
                font-size: 9px;
            }
            QListWidget::item {
                background: #1a101a;
                border-radius: 4px;
            }
        """)
        rcl.addWidget(self.batch_grid)
        
        self.loading = LoadingOverlay(rc)
        self._last_rsrc = None
        
        self.btn_stop_batch = QPushButton("STOP PROCESSING")
        self.btn_stop_batch.setFixedHeight(56)
        self.btn_stop_batch.setStyleSheet("""
            QPushButton {
                background: #ff5f40; 
                color: #fff; 
                border-radius: 18px; 
                font-weight: bold;
                margin: 0px 20px 20px 20px;
            }
            QPushButton:hover { background: #fd7b61; }
        """)
        self.btn_stop_batch.clicked.connect(self.stop_batch)
        self.btn_stop_batch.hide()
        rl.addWidget(self.btn_stop_batch)
        
        # Adjustment panel (hidden by default)
        self.adj_panel = QFrame()
        self.adj_panel.setStyleSheet("background:#1b1b1b; border-top:1px solid #2a2a2a;")
        self.adj_panel.setFixedHeight(0)
        cl.addWidget(self.adj_panel)
        
        adj_layout = QVBoxLayout(self.adj_panel)
        adj_layout.setContentsMargins(20, 10, 20, 10)
        
        self.adj_label = QLabel("Adjustment")
        self.adj_label.setStyleSheet("color:#e0e0e0; font-weight:bold;")
        adj_layout.addWidget(self.adj_label)
        
        # Adjustment layout with arrows
        adj_controls = QHBoxLayout()
        
        btn_left = QPushButton("❮")
        btn_left.setFixedSize(30, 30)
        btn_left.setStyleSheet("background:#2a2a2a; color:#fff; border-radius:15px; font-weight:bold;")
        btn_left.clicked.connect(lambda: self.adj_slider.setValue(self.adj_slider.value() - 5))
        adj_controls.addWidget(btn_left)
        
        self.adj_slider = QSlider(Qt.Horizontal)
        self.adj_slider.setRange(-100, 100)
        self.adj_slider.setValue(0)
        self.adj_slider.valueChanged.connect(self.live_preview_adjustment)
        self.adj_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2a2a2a;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ba68c8;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        adj_controls.addWidget(self.adj_slider)
        
        btn_right = QPushButton("❯")
        btn_right.setFixedSize(30, 30)
        btn_right.setStyleSheet("background:#2a2a2a; color:#fff; border-radius:15px; font-weight:bold;")
        btn_right.clicked.connect(lambda: self.adj_slider.setValue(self.adj_slider.value() + 5))
        adj_controls.addWidget(btn_right)
        
        adj_layout.addLayout(adj_controls)
        
        btn_layout = QHBoxLayout()
        
        apply_adj = GradientButton("Apply")
        apply_adj.setFixedHeight(32)
        apply_adj.clicked.connect(self.apply_adjustment)
        btn_layout.addWidget(apply_adj)
        
        cancel_adj = QPushButton("Cancel")
        cancel_adj.setFixedHeight(32)
        cancel_adj.clicked.connect(self.cancel_adjustment)
        cancel_adj.setStyleSheet("background:#333; color:#e0e0e0; border:none; border-radius:16px; font-weight:bold;")
        btn_layout.addWidget(cancel_adj)
        
        adj_layout.addLayout(btn_layout)
        
        self.current_adjustment = None
        self._adj_snapshot = None # For live preview

    def resizeEvent(self, e): 
        self.loading.resize(self.loading.parent().size())
        super().resizeEvent(e)
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space: 
            self.canvas.space_h = True
            self.canvas.setCursor(Qt.OpenHandCursor)
        elif e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_V:
            self.paste_from_clipboard()
        elif e.key() == Qt.Key_V: 
            self.set_v()
        elif e.key() == Qt.Key_C: 
            self.set_c()
        elif e.key() == Qt.Key_Escape: 
            self.set_v()
            self.hide_adjustment_panel()

    def paste_from_clipboard(self):
        mime = QApplication.clipboard().mimeData()
        if mime.hasImage():
            img = mime.imageData()
            if isinstance(img, QImage):
                img = img.convertToFormat(QImage.Format_RGB32)
                arr = np.array(img.bits()).reshape((img.height(), img.width(), 4))
                self.handle_files(cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR))
        elif mime.hasUrls() and mime.urls():
            u = mime.urls()[0]
            self.handle_files(u.toLocalFile() if u.isLocalFile() else u.toString())
            
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space: 
            self.canvas.space_h = False
            self.canvas.setCursor(Qt.ArrowCursor)

    def set_v(self): 
        self.btn_v.setChecked(True)
        self.btn_c.setChecked(False)
        self.canvas.crop_mode = False
        self.canvas.update()
        
    def set_c(self):
        if self.current_img is None: 
            self.btn_c.setChecked(False)
            return
        self.btn_v.setChecked(False)
        self.btn_c.setChecked(True)
        self.canvas.crop_mode = True
        self.canvas.update()

    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.jpeg)")
        [self.handle_files(p)] if p else None
        
    def handle_files(self, p):
        if isinstance(p, list):
            if len(p) > 0 and not (not p[0].startswith('http') and os.path.exists(p[0]) and os.path.isdir(p[0])): p = p[0]
            else: return
        if isinstance(p, str):
            if not p.startswith('http') and os.path.isdir(p): return
            self.loading_crop.show()
            self.loadw = Worker(lambda: self._load_img_bg(p))
            self.loadw.finished.connect(self._handle_files_done)
            self.loadw.start()
        elif isinstance(p, np.ndarray):
            self._handle_files_done(p)

    def _handle_files_done(self, img):
        self.loading_crop.hide()
        if img is not None:
            self.current_img = img
            self.canvas.set_image(img)
            self.stack.setCurrentIndex(1)
            h, w = img.shape[:2]
            self.inp_w.setText(str(w))
            self.inp_h.setText(str(h))
            self._sidebar_orig = img.copy()

    def reset_workspace(self):
        self.stop_batch()
        self.current_img = None
        if hasattr(self, '_last_rsrc'): self._last_rsrc = None
        if hasattr(self, '_last_sr'): delattr(self, '_last_sr')
        
        # Crop reset
        if hasattr(self, 'canvas'):
            self.canvas.cv_img = None
            self.canvas._pix = None
            if hasattr(self.canvas, '_crop_s'): self.canvas._crop_s = None
            if hasattr(self.canvas, '_crop_e'): self.canvas._crop_e = None
            self.canvas.update()
            
        if hasattr(self, 'crop_prev'): self.crop_prev.set_pix(QPixmap())
        
        # 4K reset
        if hasattr(self, 'r_src'): self.r_src.set_pix(QPixmap())
        if hasattr(self, 'r_res'): self.r_res.set_pix(QPixmap())
        if hasattr(self, 'batch_grid'): self.batch_grid.clear()
        
        self.batch_files = []
        
        # Navigation
        if hasattr(self, 'stack'):
            self.stack.setCurrentIndex(0)
            
        self.reset_sidebar_adj()
    
    def handle_rdrop(self, path_or_data):
        if isinstance(path_or_data, list):
            self.start_batch_4k(path_or_data)
        elif isinstance(path_or_data, str):
            if os.path.isdir(path_or_data):
                self.start_batch_4k([path_or_data])
            else:
                self.loading.show()
                self.rload = Worker(lambda: self._load_img_bg(path_or_data))
                self.rload.finished.connect(self._handle_rdrop_done)
                self.rload.start()
        else: self._handle_rdrop_done(path_or_data)

    def start_batch_4k(self, paths):
        img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
        self.batch_files = []
        for p in paths:
            if not isinstance(p, str): continue
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if not isinstance(f, str): continue
                    if os.path.splitext(f)[1].lower() in img_exts:
                        self.batch_files.append(os.path.join(p, f))
            elif os.path.splitext(p)[1].lower() in img_exts:
                self.batch_files.append(p)
                
        if not self.batch_files:
            QMessageBox.warning(self, "No Images", "No valid images found for batch processing.")
            return
            
        out_dir = QFileDialog.getExistingDirectory(self, f"Select Output Dir for {len(self.batch_files)} images")
        if not out_dir: return
        
        self.batch_out_dir = out_dir
        self.batch_index = 0
        
        self.batch_grid.clear()
        for p in self.batch_files:
            item = QListWidgetItem(os.path.basename(p))
            self.batch_grid.addItem(item)
            
        self.batch_running = True
        self.btn_stop_batch.show()
        self.loading.show()
        self.process_next_batch_item()

    def process_next_batch_item(self):
        if not self.batch_running or self.batch_index >= len(self.batch_files):
            self.batch_running = False
            self.btn_stop_batch.hide()
            self.loading.hide()
            self.r_src.set_pix(QPixmap())
            self.r_res.set_pix(QPixmap())
            QMessageBox.information(self, "Batch Complete", f"Successfully enhanced {len(self.batch_files)} images!")
            return
            
        item = self.batch_grid.item(self.batch_index)
        if item:
            item.setText(f"[PROCESSING] {os.path.basename(self.batch_files[self.batch_index])}")
            self.batch_grid.scrollToItem(item)
            
        p = self.batch_files[self.batch_index]
        self.rload = Worker(lambda: self._load_img_bg(p))
        self.rload.finished.connect(self._handle_batch_loaded)
        self.rload.start()
        
    def _handle_batch_loaded(self, cv_img):
        if cv_img is None:
            self.batch_index += 1
            self.process_next_batch_item()
            return
            
        h, w = cv_img.shape[:2]
        max_s = 600
        if w > max_s or h > max_s:
            sc = max_s/max(w,h)
            prev = cv2.resize(cv_img, (0,0), fx=sc, fy=sc)
        else: prev = cv_img
        ph, pw = prev.shape[:2]
        qi = QImage(prev.tobytes(), pw, ph, 3*pw, QImage.Format_BGR888)
        self.r_src.set_pix(QPixmap.fromImage(qi))
        
        self.worker = Worker(lambda: self.proc.apply_super_res(cv_img))
        self.worker.finished.connect(self._handle_batch_sr_done)
        self.worker.start()
        
    def _handle_batch_sr_done(self, res):
        if res is not None:
            p = self.batch_files[self.batch_index]
            base_name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(self.batch_out_dir, f"{base_name}_4K.jpg")
            is_success, buffer = cv2.imencode(".jpg", res, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if is_success:
                with open(out_path, "wb") as f:
                    f.write(buffer)
            
            h, w = res.shape[:2]
            max_s = 600
            if w > max_s or h > max_s:
                sc = max_s/max(w,h)
                prev = cv2.resize(res, (0,0), fx=sc, fy=sc)
            else: prev = res
            ph, pw = prev.shape[:2]
            qi = QImage(prev.tobytes(), pw, ph, 3*pw, QImage.Format_BGR888)
            self.r_res.set_pix(QPixmap.fromImage(qi))
            
            # Update grid item
            item = self.batch_grid.item(self.batch_index)
            if item:
                item.setText(f"[DONE] {os.path.basename(p)}")
                item.setIcon(QIcon(QPixmap.fromImage(qi)))
                
        self.batch_index += 1
        self.process_next_batch_item()

    def _load_img_bg(self, p):
        try:
            if p.startswith('http://') or p.startswith('https://'):
                import urllib.request
                req = urllib.request.Request(p, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()
            else:
                with open(p, 'rb') as f: data = f.read()
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print("Load Error", e)
            return None

    def _handle_rdrop_done(self, cv_img):
        self.loading.hide()
        if cv_img is not None:
            self._last_rsrc = cv_img
            # PRE-SCALE PREVIEW TO PREVENT HANG
            h, w = cv_img.shape[:2]
            max_s = 600
            if w > max_s or h > max_s:
                sc = max_s/max(w,h)
                prev_img = cv2.resize(cv_img, (0,0), fx=sc, fy=sc)
            else: prev_img = cv_img
            
            ph, pw = prev_img.shape[:2]
            qi = QImage(prev_img.tobytes(), pw, ph, 3*pw, QImage.Format_BGR888)
            self.r_src.set_pix(QPixmap.fromImage(qi))

    def run_4k(self, img):
        if img is None: return
        self.loading.show()
        self.worker = Worker(lambda: self.proc.apply_super_res(img))
        self.worker.finished.connect(self._sr_done)
        self.worker.start()
        
    def _sr_done(self, res):
        self.loading.hide()
        if res is not None:
            self._last_sr = res
            # Pre-scale result for preview label
            h,w = res.shape[:2]
            max_s = 600
            if w > max_s or h > max_s:
                sc = max_s/max(w,h)
                prev_img = cv2.resize(res, (0,0), fx=sc, fy=sc)
            else: prev_img = res
            ph, pw = prev_img.shape[:2]
            qi = QImage(prev_img.tobytes(), pw, ph, 3*pw, QImage.Format_BGR888)
            self.r_res.set_pix(QPixmap.fromImage(qi))

    def save_sr(self):
        if hasattr(self, '_last_sr'):
            p, _ = QFileDialog.getSaveFileName(self, "Save 4K", "upscaled.jpg", "JPEG Image (*.jpg);;PNG Image (*.png)")
            if p:
                ext = ".jpg" if p.endswith(".jpg") else ".png"
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 90] if ext == ".jpg" else []
                cv2.imencode(ext, self._last_sr, params)[1].tofile(p)

    def upd_crop_prev(self, pix): 
        self.crop_prev.set_pix(pix)
        
    def confirm_crop(self):
        if not self.canvas._crop_s or not self.canvas._crop_e: return
        r = QRect(self.canvas._crop_s, self.canvas._crop_e).normalized()
        p1, p2 = self.canvas.widget_to_image(r.topLeft()), self.canvas.widget_to_image(r.bottomRight())
        res = self.proc.crop(self.current_img, p1.x(), p1.y(), p2.x(), p2.y())
        if res is not None and self.current_img is not None: 
            self.history.append(self.current_img.copy())
            self.current_img = res
            self.canvas.set_image(res)
            h,w = res.shape[:2]
            self.inp_w.setText(str(w))
            self.inp_h.setText(str(h))
        self.set_v()

    def apply_res(self):
        try:
            w, h = int(self.inp_w.text()), int(self.inp_h.text())
            if self.current_img is not None and w > 0: 
                self.history.append(self.current_img.copy())
                self.current_img = self.proc.resize(self.current_img, w, h)
                self.canvas.set_image(self.current_img)
        except: 
            pass
            
    def undo(self):
        if self.history: 
            self.redo_s.append(self.current_img.copy())
            self.current_img = self.history.pop()
            self.canvas.set_image(self.current_img)
            
    def redo(self):
        if self.redo_s: 
            self.history.append(self.current_img.copy())
            self.current_img = self.redo_s.pop()
            self.canvas.set_image(self.current_img)
            
    def save_edit(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save", "result.png", "PNG (*.png)")
        [cv2.imencode(".png", self.current_img)[1].tofile(p)] if p else None
        
    def on_ratio(self, i): 
        ratios = [None, 1.0, 16/9, 4/3]
        self.canvas.aspect_ratio = ratios[i]
        # Reset crop when ratio changes
        if self.canvas.crop_mode:
            self.canvas._crop_s = None
            self.canvas._crop_e = None
            self.canvas.update()
    
    def show_brightness(self):
        self.show_adjustment_panel("BRIGHTNESS", -100, 100, "brightness")
        
    def show_contrast(self):
        self.show_adjustment_panel("CONTRAST", -50, 50, "contrast")
        
    def show_saturation(self):
        self.show_adjustment_panel("SATURATION", 0, 200, "saturation")
        
    def show_rotation(self):
        self.show_adjustment_panel("ROTATION", -180, 180, "rotation")
        
    def show_blur(self):
        self.show_adjustment_panel("BLUR", 0, 50, "blur")
    
    def show_adjustment_panel(self, label, min_val, max_val, adj_type):
        if self.current_img is None:
            return
            
        self._adj_snapshot = self.current_img.copy()
        self.adj_label.setText(label)
        
        # Block signals temporarily to prevent early updates
        self.adj_slider.blockSignals(True)
        self.adj_slider.setRange(min_val, max_val)
        self.adj_slider.setValue(0 if adj_type != "saturation" else 100)
        self.adj_slider.blockSignals(False)
        
        self.current_adjustment = adj_type
        
        # Animate panel
        self.adj_panel.setFixedHeight(120)
        
    def hide_adjustment_panel(self):
        self.adj_panel.setFixedHeight(0)
        self.current_adjustment = None
        
    def sidebar_adj_br(self, v): self.apply_sidebar_live()
    def sidebar_adj_ct(self, v): self.apply_sidebar_live()
    def sidebar_adj_st(self, v): self.apply_sidebar_live()
    def sidebar_adj_rt(self, v): self.apply_sidebar_live()
    def sidebar_adj_bl(self, v): self.apply_sidebar_live()

    def live_preview_adjustment(self):
        if self._adj_snapshot is None or self.current_adjustment is None:
            return
        val = self.adj_slider.value()
        img = self._adj_snapshot.copy()
        if self.current_adjustment == "brightness": img = self.proc.adjust_brightness(img, val)
        elif self.current_adjustment == "contrast": img = self.proc.adjust_contrast(img, val)
        elif self.current_adjustment == "saturation": img = self.proc.adjust_saturation(img, val)
        elif self.current_adjustment == "rotation": img = self.proc.rotate(img, val)
        elif self.current_adjustment == "blur": img = self.proc.blur(img, val)
        if img is not None:
            self.canvas.set_image(img)
            self.current_img = img

    def cancel_adjustment(self):
        if self._adj_snapshot is not None:
            self.current_img = self._adj_snapshot.copy()
            self.canvas.set_image(self.current_img)
        self.hide_adjustment_panel()

    def apply_adjustment(self):
        if self._adj_snapshot is not None:
            self.history.append(self._adj_snapshot.copy())
            self._adj_snapshot = None
        self.hide_adjustment_panel()

    def apply_sidebar_live(self):
        if not hasattr(self, '_sidebar_orig') or self._sidebar_orig is None: return
        img = self._sidebar_orig.copy()
        
        b = self.sb_br.value()
        c = self.sb_ct.value()
        s = self.sb_st.value()
        r = self.sb_rt.value()
        bl = self.sb_bl.value()
        
        if b != 0: img = self.proc.adjust_brightness(img, b)
        if c != 0: img = self.proc.adjust_contrast(img, c)
        if s != 100: img = self.proc.adjust_saturation(img, s)
        if r != 0: img = self.proc.rotate(img, r)
        if bl != 0: img = self.proc.blur(img, bl)
        
        self.canvas.set_image(img)
        self.current_img = img

    def commit_sidebar_adj(self):
        if hasattr(self, '_sidebar_orig') and self._sidebar_orig is not None:
            self.history.append(self._sidebar_orig.copy())
            self._sidebar_orig = self.current_img.copy()
            # Reset sliders to neutral without triggering live (block signals)
            for sb in [self.sb_br, self.sb_ct, self.sb_st, self.sb_rt, self.sb_bl]:
                sb.blockSignals(True)
                sb.setValue(0 if sb != self.sb_st else 100)
                # Manually update value labels since signals are blocked
                # We need to find the label in the parent row
                sb.blockSignals(False)

    def reset_sidebar_adj(self):
        if hasattr(self, '_sidebar_orig'):
            self.current_img = self._sidebar_orig.copy()
            self.canvas.set_image(self.current_img)
            for sb in [self.sb_br, self.sb_ct, self.sb_st, self.sb_rt, self.sb_bl]:
                sb.blockSignals(True)
                sb.setValue(0 if sb != self.sb_st else 100)
                sb.blockSignals(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    icon_path = os.path.join(BASE_DIR, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    win = CropToolWindow()
    
    # Also set the window icon directly on the main window just in case
    if os.path.exists(icon_path):
        win.setWindowIcon(QIcon(icon_path))
        
    win.show()
    sys.exit(app.exec())