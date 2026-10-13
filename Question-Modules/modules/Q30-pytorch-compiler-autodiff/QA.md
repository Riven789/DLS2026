# Q30 QA and review record

- **Author:** Kukla
- **Question:** How does the PyTorch compiler work (layers)? Why does functional AD use more memory than tape-based?
- **Survey source:** #29
- **Status:** author complete; peer review pending.

## Author self-check
- [x] note.tex builds (3pp) w/ citations + bibliography.
- [x] deck.tex builds, 5 content slides (<=10).
- [x] discussion (5), projects (3), quiz (1+answer) build.
- [x] code/autodiff_memory.py runs (stdlib); labeled memory-accounting SIMULATION, not real autograd.
- [x] Claims calibrated: torch.compile stack layers named precisely (Dynamo/AOTAutograd/Inductor/Triton); functional-AD memory claim qualified ('naive formulation', not 'always').
