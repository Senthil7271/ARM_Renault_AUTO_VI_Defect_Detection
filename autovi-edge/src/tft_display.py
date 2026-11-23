"""AutoVI Edge: 1.8" ST7735 Physical Indicator Display Driver.

Provides real-time physical visual indication (Vibrant Green for PASS, Bright Red for REJECT)
synchronized with Hailo-8 and ARM Cortex-A76 inspection inference.
"""

import logging
import time
from typing import Optional
from PIL import Image, ImageDraw

logger = logging.getLogger("AutoVI-TFT")

class IndustrialTFT:
    def __init__(self, dc_pin: int = 24, rst_pin: int = 25, width: int = 160, height: int = 128):
        self.width = width
        self.height = height
        self.is_active = False

        try:
            import spidev
            from gpiozero import DigitalOutputDevice

            self.dc = DigitalOutputDevice(dc_pin)
            self.rst = DigitalOutputDevice(rst_pin)

            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 24000000
            self.spi.mode = 0

            self._reset()
            self._init_display()
            self.is_active = True
            logger.info("1.8\" ST7735 TFT Display initialized successfully.")
            self.show_ready()
        except Exception as e:
            logger.warning(f"TFT Display hardware not available or disabled: {e}")
            self.is_active = False

    def _command(self, cmd: int):
        self.dc.off()
        self.spi.writebytes([cmd])

    def _data(self, val):
        self.dc.on()
        if isinstance(val, int):
            self.spi.writebytes([val])
        else:
            self.spi.writebytes2(val)

    def _reset(self):
        self.rst.on(); time.sleep(0.01)
        self.rst.off(); time.sleep(0.01)
        self.rst.on(); time.sleep(0.05)

    def _init_display(self):
        self._command(0x01); time.sleep(0.12)  # Software Reset
        self._command(0x11); time.sleep(0.12)  # Sleep Out
        self._command(0x3A); self._data(0x05)   # 16-bit RGB565
        self._command(0x36); self._data(0xA0)   # Mode 1: Landscape (160x128) True RGB
        self._command(0x20)                    # Inversion OFF
        self._command(0x13); time.sleep(0.01)  # Normal Display Mode
        self._command(0x29); time.sleep(0.05)  # Display ON

    def display(self, image: Image.Image):
        if not self.is_active:
            return

        try:
            img = image.convert("RGB").resize((self.width, self.height))
            raw = img.tobytes()
            buf = bytearray(self.width * self.height * 2)

            j = 0
            for i in range(0, len(raw), 3):
                r, g, b = raw[i], raw[i+1], raw[i+2]
                rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                buf[j] = (rgb565 >> 8) & 0xFF
                buf[j+1] = rgb565 & 0xFF
                j += 2

            self._command(0x2A); self._data([0x00, 0x00, 0x00, self.width - 1])
            self._command(0x2B); self._data([0x00, 0x00, 0x00, self.height - 1])
            self._command(0x2C)
            self._data(buf)
        except Exception as e:
            logger.error(f"Error rendering to TFT: {e}")

    def show_ready(self):
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.width, self.height), fill=(15, 23, 42))  # Dark Blue Slate
        draw.rectangle((0, 0, self.width, 22), fill=(30, 41, 59))
        draw.text((10, 5), "AutoVI Edge Lab", fill=(56, 189, 248))
        draw.text((25, 48), "SYSTEM READY", fill=(255, 255, 255))
        draw.text((15, 72), "Hailo-8 + ExecuTorch", fill=(148, 163, 184))
        draw.rectangle((0, self.height - 22, self.width, self.height), fill=(10, 15, 30))
        draw.text((10, self.height - 18), "Awaiting Inspection", fill=(100, 116, 139))
        self.display(img)

    def show_verdict(self, verdict: str, category: str, score: float, threshold: float):
        if not self.is_active:
            return

        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        is_pass = (verdict.upper() == "PASS")
        if is_pass:
            bg = (0, 190, 50)           # Pure Bright Green
            title = "PASS [OK]"
            sub = "NOMINAL CAD"
        else:
            bg = (240, 10, 10)          # Pure Bright Red
            title = "REJECT [NG]"
            sub = "DEFECT DETECTED"

        # Edge-to-Edge Fill
        draw.rectangle((0, 0, self.width, self.height), fill=bg)

        # Header Bar
        draw.rectangle((0, 0, self.width, 22), fill=(20, 20, 20))
        draw.text((8, 5), f"AutoVI: {category.upper()}", fill=(255, 255, 255))

        # Main Decision
        draw.text((22, 45), title, fill=(255, 255, 255))
        draw.text((25, 70), sub, fill=(240, 240, 240))

        # Footer Telemetry
        draw.rectangle((0, self.height - 24, self.width, self.height), fill=(15, 15, 15))
        draw.text((8, self.height - 19), f"Score: {score:.2f} | Tau: {threshold:.2f}", fill=(220, 220, 220))

        self.display(img)


# Global instance
_tft_instance: Optional[IndustrialTFT] = None

def get_tft() -> IndustrialTFT:
    global _tft_instance
    if _tft_instance is None:
        _tft_instance = IndustrialTFT()
    return _tft_instance
