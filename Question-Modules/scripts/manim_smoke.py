"""Manim smoke test for the course pipeline: animated attention diagram.

One scene: tokens flowing into an attention block, weights visualized as
lines whose thickness = attention weight. Proves diagrams + animation render.
"""
from manim import *

BG = "#1B2A4A"      # course ink navy
CREAM = "#FAF6EE"
SKY = "#D9E4F5"
ACCENT = "#C97B4A"  # terracotta, brightened for video
MONO = "DejaVu Sans Mono"


class AttentionSmoke(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Attention, visually", font_size=44, color=CREAM, font=MONO)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title), run_time=1.2)
        self.wait(0.5)

        # token boxes
        tokens = ["The", "robot", "lifted", "the", "box"]
        boxes = VGroup()
        for t in tokens:
            box = VGroup(
                Rectangle(width=1.5, height=0.7, color=SKY, stroke_width=2),
                Text(t, font_size=22, color=CREAM, font=MONO),
            )
            boxes.add(box)
        boxes.arrange(RIGHT, buff=0.5).move_to(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.4) for b in boxes], lag_ratio=0.12), run_time=1.6)
        self.wait(0.8)

        # query token highlighted
        q = boxes[1]
        self.play(q.animate.set_stroke(color=ACCENT, width=4), run_time=0.8)
        qlab = Text("query: 'robot'", font_size=22, color=ACCENT, font=MONO)
        qlab.next_to(q, UP, buff=0.4)
        self.play(FadeIn(qlab), run_time=0.6)
        self.wait(0.6)

        # attention weights as arcs to each key
        weights = [0.05, 0.10, 0.45, 0.15, 0.25]
        arcs = VGroup()
        labels = VGroup()
        for i, (b, w) in enumerate(zip(boxes, weights)):
            if i == 1:
                continue
            arc = CurvedArrow(q.get_bottom(), b.get_top(), angle=-TAU / 4,
                              color=SKY, stroke_width=max(1.5, 10 * w), tip_length=0.15)
            arcs.add(arc)
            lab = Text(f"{w:.2f}", font_size=18, color=SKY, font=MONO)
            lab.next_to(arc.point_from_proportion(0.5), DOWN, buff=0.15)
            labels.add(lab)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=2.0)
        self.play(FadeIn(labels), run_time=0.6)
        self.wait(1.0)

        # weighted blend -> contextual embedding
        out = VGroup(
            Rectangle(width=2.6, height=0.8, color=ACCENT, stroke_width=3),
            Text("contextual embedding", font_size=20, color=CREAM, font=MONO),
        )
        out.move_to(DOWN * 2.4)
        arrow = Arrow(boxes.get_bottom() + DOWN * 0.15, out.get_top() + UP * 0.1,
                      color=ACCENT, stroke_width=3)
        self.play(GrowArrow(arrow), run_time=0.8)
        self.play(FadeIn(out, shift=UP * 0.3), run_time=0.8)
        self.wait(1.5)

        cap = Text("weights sum to 1 · softmax over similarities", font_size=20,
                   color=SKY, font=MONO)
        cap.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
