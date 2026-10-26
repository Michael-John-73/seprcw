# seprcw
Stable Embedding and Pilot-Guided Recovery of Cropped Block Watermarks

## Usage

`seprcw.embed(img, bits)` embeds a 12-bit payload (QIM on 8x8 DCT blocks, tiled with pilot blocks); `seprcw.extract(cropped)` finds the crop offset via the pilots and returns `(bits, offsets, pilot_match)`. Requires numpy; tests: `pytest tests`.
