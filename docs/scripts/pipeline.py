"""README 처리 과정 그림(docs/pipeline.png)을 만든다 — 노드가 하는 단계를 왼쪽에서 오른쪽으로.

실행: python docs/scripts/pipeline.py   (Pillow, 한글 글꼴 맑은 고딕 또는 Noto Sans CJK 필요)
결과 그림은 아직 넣지 않았다(이 플러그인은 돌려 보지 않았다). 단계 글은 아래 STEPS 하나만 고친다.
"""
import os

from PIL import Image, ImageDraw, ImageFont

# (한국어, 영어, 칸 종류) — 칸 종류: in 입력 · step 노드 안 처리 · out 출력
STEPS = [
    ('가사 · 스타일 태그', 'Lyrics + style tags', 'in'),
    ('HeartMuLa 3B', 'audio tokens, 12.5 Hz', 'step'),
    ('HeartCodec', 'tokens to 48 kHz', 'step'),
    ('노래 WAV', '48 kHz stereo song', 'out'),
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
COLORS = {"in": ("#e8f1fb", "#3b7dd8"), "step": ("#f3eefc", "#7b5cd6"), "out": ("#e9f7ef", "#2f9e5f")}
BOX_W, BOX_H, GAP, PAD = 250, 120, 56, 28


def font(size):
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("한글 글꼴을 찾지 못했다 — FONT_CANDIDATES 에 경로를 더한다")


def main():
    n = len(STEPS)
    w = PAD * 2 + n * BOX_W + (n - 1) * GAP
    h = PAD * 2 + BOX_H
    img = Image.new("RGB", (w * 2, h * 2), "#ffffff")  # 2배로 그려 글자를 또렷하게
    d = ImageDraw.Draw(img)
    ko, en = font(50), font(34)
    for i, (k, e, kind) in enumerate(STEPS):
        x = (PAD + i * (BOX_W + GAP)) * 2
        y = PAD * 2
        fill, line = COLORS[kind]
        d.rounded_rectangle([x, y, x + BOX_W * 2, y + BOX_H * 2], radius=28, fill=fill, outline=line, width=4)
        cx = x + BOX_W
        d.text((cx, y + BOX_H - 4), k, font=ko, fill="#1f2328", anchor="ms")
        d.text((cx, y + BOX_H + 54), e, font=en, fill="#57606a", anchor="ms")
        if i < n - 1:
            ax = x + BOX_W * 2 + 14
            ay = y + BOX_H
            d.line([ax, ay, ax + GAP * 2 - 40, ay], fill="#8c959f", width=6)
            d.polygon([(ax + GAP * 2 - 28, ay), (ax + GAP * 2 - 48, ay - 14), (ax + GAP * 2 - 48, ay + 14)], fill="#8c959f")
    img = img.resize((w, h), Image.LANCZOS)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.png")
    img.save(out, optimize=True)
    print(out, img.size)


if __name__ == "__main__":
    main()
