"""
Procedural achievement badge rendering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app_paths import achievement_badges_dir
from achievement_catalog import AchievementDefinition


RENDERER_VERSION = 2
FALLBACK_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00"
    b"\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01z"
    b"^\xab?\x00\x00\x00\x00IEND\xaeB`\x82"
)

PALETTES = [
    ("#2D6B6B", "#72C9BE", "#FFF5D6"),
    ("#8E5CF7", "#DCC6FF", "#FFFDF7"),
    ("#C58C2A", "#E7C873", "#FFF6D8"),
    ("#2F7D4F", "#9ED6B1", "#F1FFF5"),
    ("#B23B35", "#F0A29E", "#FFF2EF"),
    ("#374151", "#A7B0BE", "#F8FAFC"),
]

RARITY_ACCENT = {
    "common": "#8A8178",
    "uncommon": "#2F7D4F",
    "rare": "#2D6B6B",
    "epic": "#8E5CF7",
    "legendary": "#C58C2A",
    "mythic": "#B23B35",
    "secret": "#26211D",
}


class AchievementBadgeRenderer:
    """Render deterministic PNG badges for achievement definitions."""

    def __init__(self, badges_dir: Path | None = None):
        self.badges_dir = badges_dir or achievement_badges_dir()
        self.badges_dir.mkdir(parents=True, exist_ok=True)

    def badge_path(self, definition: AchievementDefinition, *, unlocked: bool = True) -> Path:
        suffix = "unlocked" if unlocked else "locked"
        return self.badges_dir / f"v{RENDERER_VERSION}-{definition.id}-{suffix}.png"

    def render_badge(self, definition: AchievementDefinition, *, unlocked: bool = True, size: int = 192) -> Path:
        path = self.badge_path(definition, unlocked=unlocked)
        if path.exists():
            return path
        try:
            from PyQt6.QtCore import QPointF, QRectF, Qt
            from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen, QPolygonF
        except Exception:
            path.write_bytes(FALLBACK_PNG)
            return path
        if QGuiApplication.instance() is None:
            path.write_bytes(FALLBACK_PNG)
            return path

        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)

        palette = self._palette(definition)
        primary = QColor(palette[0])
        secondary = QColor(palette[1])
        background = QColor(palette[2])
        accent = QColor(RARITY_ACCENT.get(definition.rarity, "#2D6B6B"))
        if not unlocked:
            primary = QColor("#9B948C")
            secondary = QColor("#D8D0C2")
            background = QColor("#F0E9DE")
            accent = QColor("#B8AEA2")

        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)

            rect = QRectF(10, 10, size - 20, size - 20)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, 34, 34)

            painter.setBrush(primary)
            painter.drawEllipse(QRectF(28, 28, size - 56, size - 56))
            painter.setBrush(secondary)
            painter.drawEllipse(QRectF(42, 42, size - 84, size - 84))

            painter.setPen(QPen(accent, 8))
            painter.drawArc(QRectF(32, 32, size - 64, size - 64), 210 * 16, -300 * 16)
            painter.setPen(Qt.PenStyle.NoPen)

            self._draw_family(painter, definition, primary, accent, size, QPointF, QPolygonF, QRectF, Qt, QPen)
            self._draw_motif(painter, definition, accent, size, QRectF, Qt, QPen)

            painter.setPen(QColor("#26211D") if unlocked else QColor("#6F665E"))
            font = QFont("Arial", 22)
            font.setBold(True)
            painter.setFont(font)
            initials = "".join(part[:1] for part in definition.title.split()[:2]).upper()[:2]
            painter.drawText(QRectF(0, size - 54, size, 30), Qt.AlignmentFlag.AlignCenter, initials or "?")
        except Exception:
            if painter.isActive():
                painter.end()
            path.write_bytes(FALLBACK_PNG)
            return path
        finally:
            if painter.isActive():
                painter.end()
        image.save(str(path), "PNG")
        return path

    def _palette(self, definition: AchievementDefinition) -> tuple[str, str, str]:
        digest = hashlib.sha256(f"{definition.id}:{definition.badge_seed}".encode("utf-8")).digest()
        return PALETTES[digest[0] % len(PALETTES)]

    def _draw_family(
        self,
        painter,
        definition: AchievementDefinition,
        primary,
        accent,
        size: int,
        QPointF,
        QPolygonF,
        QRectF,
        Qt,
        QPen,
    ):
        family = str(definition.badge_family or "trophy").lower()
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)

        if family in {"trophy", "medal", "ribbon"}:
            painter.drawRoundedRect(72, 64, 48, 46, 10, 10)
            painter.drawRect(86, 108, 20, 18)
            painter.drawRoundedRect(66, 126, 60, 12, 6, 6)
            if family != "medal":
                painter.drawEllipse(50, 70, 26, 22)
                painter.drawEllipse(116, 70, 26, 22)
            if family == "ribbon":
                painter.setBrush(primary)
                painter.drawPolygon(QPolygonF([QPointF(78, 138), QPointF(94, 162), QPointF(102, 138)]))
                painter.drawPolygon(QPolygonF([QPointF(98, 138), QPointF(114, 162), QPointF(122, 138)]))
            return

        if family in {"waveform", "microphone"}:
            if family == "microphone":
                painter.drawRoundedRect(78, 54, 36, 66, 18, 18)
                painter.setPen(QPen(accent, 8))
                painter.drawArc(QRectF(60, 78, 72, 60), 200 * 16, 140 * 16)
                painter.drawLine(96, 138, 96, 158)
                painter.drawLine(78, 158, 114, 158)
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                for idx, height in enumerate([28, 52, 82, 58, 34]):
                    painter.drawRoundedRect(QRectF(54 + idx * 18, 104 - height / 2, 10, height), 5, 5)
            return

        if family in {"window", "stack"}:
            painter.drawRoundedRect(52, 58, 88, 76, 10, 10)
            painter.setBrush(primary)
            painter.drawRect(52, 78, 88, 8)
            if family == "stack":
                painter.setBrush(accent)
                painter.drawRoundedRect(62, 72, 88, 76, 10, 10)
            return

        if family in {"clock", "calendar"}:
            painter.drawEllipse(56, 56, 80, 80)
            painter.setPen(QPen(primary, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(96, 96, 96, 68)
            painter.drawLine(96, 96, 116, 108)
            painter.setPen(Qt.PenStyle.NoPen)
            return

        if family == "lightning":
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(104, 44),
                        QPointF(68, 104),
                        QPointF(96, 104),
                        QPointF(82, 150),
                        QPointF(128, 84),
                        QPointF(100, 84),
                    ]
                )
            )
            return

        if family == "dictionary":
            painter.drawRoundedRect(58, 54, 76, 94, 8, 8)
            painter.setBrush(primary)
            painter.drawRect(72, 54, 8, 94)
            return

        painter.drawEllipse(62, 62, 68, 68)

    def _draw_motif(self, painter, definition: AchievementDefinition, accent, size: int, QRectF, Qt, QPen):
        motif = str(definition.badge_motif or "").lower()
        painter.setPen(QPen(accent, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if motif in {"spark", "orbit", "bolt"}:
            painter.drawLine(142, 42, 142, 66)
            painter.drawLine(130, 54, 154, 54)
        elif motif in {"book", "typebar", "stack"}:
            painter.drawLine(56, 150, 136, 150)
            painter.drawLine(70, 160, 122, 160)
        elif motif in {"wave", "browser", "chat", "docs", "editor", "email"}:
            for idx in range(3):
                painter.drawArc(QRectF(50 + idx * 20, 138, 32, 20), 20 * 16, 140 * 16)
        elif motif in {"window", "cursor"}:
            painter.drawLine(142, 132, 160, 154)
            painter.drawLine(160, 154, 150, 154)
        else:
            painter.drawEllipse(138, 138, 18, 18)
        painter.setPen(Qt.PenStyle.NoPen)
