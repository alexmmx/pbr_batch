# pbr_batch_tool

Batch-generate **PBR texture maps** (Albedo, Height, Normal, Roughness, AO, Metallic, ORM) from **seamless PNG textures**.  
Preserves seamless tiling using wrap-padding. Supports ZIP export.

---

## ✨ Features
- Input: seamless PNG (1K–4K recommended, neutral lighting, no baked shadows).
- Outputs:
  - `*_albedo.png`
  - `*_height.png` (grayscale from luminance)
  - `*_normal.png` (tangent space)
  - `*_roughness.png` (from local contrast)
  - `*_ao.png` (ambient occlusion approximation)
  - `*_metallic.png` (constant map)
  - `*_ORM.png` (optional, R=AO, G=Roughness, B=Metallic)
- Recursive folder processing, mirrored structure in `out/`.
- Optional: ZIP archives (one per folder and/or global).
- CLI parameters to tune Normal/AO/Roughness intensities.

---

## 📦 Installation

Requires **Python 3.9+** (tested on Linux, MacOS, Windows WSL).  

```bash
# Clone or copy repo
git clone https://github.com/yourname/pbr_batch_tool.git
cd pbr_batch_tool

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import numpy, cv2; print('numpy:', numpy.__version__, 'opencv:', cv2.__version__)"

