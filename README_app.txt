CK3Fit - photos of a real person -> Crusader Kings III Ruler Designer DNA
(made for CK3 1.19 with the A Game of Thrones mod)

HOW TO USE
1. Make a folder with photos of one person (jpg, png or webp). For several people, make one folder per
   person inside a main folder and pick the main folder. The folder name becomes the file name.
   Works best with 4 or more clear front photos and at least one turned (three-quarter) photo.
2. Open CK3Fit.exe, pick the photo folder and choose Woman or Man.
3. Press "Check photos". It tells you if a photo is blurry, too small, has an open mouth, or doesn't seem
   to match the others, and whether there are enough photos.
4. Press "Make DNA". Each person gets <name>.txt in the DNA folder. With one person it is also copied to
   the clipboard.
5. In the game: Ruler Designer -> Paste DNA, then paste the text.

The first run is slower: it loads the models and downloads two face-recognition model files (about 450 MB
in all, from the InsightFace project on GitHub) into data\results\arcface. After that it works offline.
It uses an NVIDIA graphics card if one is there, otherwise the processor (slower, but works).
Keep the "data" folder next to CK3Fit.exe.

Command line (optional):
   CK3Fit.exe --photos "C:\path\to\photos" [--man] [--out "C:\path\to\output"] [--check]

WHAT'S INSIDE / LICENCES
- No game files, no game images and no photos of people are included. The app only holds numbers learned
  from the game (slider models, colour tables, hair/beard/brow tables).
- DINOv2 (Meta) - Apache 2.0.
- MediaPipe face landmarker (Google) - Apache 2.0.
- PyTorch, transformers, NumPy, OpenCV and the other libraries - their own open-source licences.
- InsightFace face-recognition models (buffalo_l, antelopev2) - provided by InsightFace for non-commercial
  research use only. They are not included; the app downloads them from InsightFace on first run.
  Because of this, CK3Fit as a whole is for personal, non-commercial use only.
- Crusader Kings III is (c) Paradox Interactive; A Game of Thrones mod by its authors. This is an
  unofficial fan tool, not affiliated with either.
