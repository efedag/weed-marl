import glob
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile

CROP_RGB = np.array([0, 255, 0], dtype=np.uint8)
WEED_RGB = np.array([255, 0, 0], dtype=np.uint8)

# 0 = soil, 1 = crop, 2 = weed
CLASS_SOIL = 0
CLASS_CROP = 1
CLASS_WEED = 2


def load_gt_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".png":
        arr = np.array(Image.open(path))
    elif ext in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
    else:
        raise ValueError(f"Unsupported ground truth extension: {path}")

    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got {arr.shape} for {path}")
    return arr.astype(np.uint8)


def rgb_to_class_map(rgb: np.ndarray) -> np.ndarray:
    class_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
    class_map[np.all(rgb == CROP_RGB, axis=-1)] = CLASS_CROP
    class_map[np.all(rgb == WEED_RGB, axis=-1)] = CLASS_WEED
    return class_map


def convert_to_grid64(
    class_map: np.ndarray, grid_size: int = 64, weed_threshold: float = 0.05
) -> np.ndarray:
    h, w = class_map.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    if cell_h == 0 or cell_w == 0:
        raise ValueError(f"Image too small ({h}x{w}) for {grid_size}x{grid_size} grid.")

    trimmed = class_map[: cell_h * grid_size, : cell_w * grid_size]
    cells = trimmed.reshape(grid_size, cell_h, grid_size, cell_w).transpose(0, 2, 1, 3)
    flat_cells = cells.reshape(grid_size, grid_size, cell_h * cell_w)

    weed_ratio = np.mean(flat_cells == CLASS_WEED, axis=-1)
    crop_ratio = np.mean(flat_cells == CLASS_CROP, axis=-1)

    grid = np.full((grid_size, grid_size), CLASS_SOIL, dtype=np.uint8)
    grid[crop_ratio > 0] = CLASS_CROP
    grid[weed_ratio >= weed_threshold] = CLASS_WEED
    return grid


def save_preview(grid: np.ndarray, out_png: Path) -> None:
    palette = {
        CLASS_SOIL: np.array([30, 30, 30], dtype=np.uint8),
        CLASS_CROP: np.array([40, 180, 75], dtype=np.uint8),
        CLASS_WEED: np.array([220, 50, 50], dtype=np.uint8),
    }
    vis = np.zeros((*grid.shape, 3), dtype=np.uint8)
    for k, color in palette.items():
        vis[grid == k] = color
    scale = 6
    img = Image.fromarray(vis, mode="RGB").resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), resample=Image.Resampling.NEAREST
    )
    img.save(out_png)


def main() -> None:
    rededge_root = Path(r"C:\Users\ssozy\Downloads\Orthomosaic\Orthomosaic\RedEdge")
    out_dir = Path("data/grids_64")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(
        set(
            glob.glob(str(rededge_root / "**/*_gt.png"), recursive=True)
            + glob.glob(str(rededge_root / "**/*_gt.tif"), recursive=True)
            + glob.glob(str(rededge_root / "**/*_gt.tiff"), recursive=True)
        )
    )

    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found under {rededge_root}")

    for idx, file_str in enumerate(gt_files):
        gt_path = Path(file_str)
        rgb = load_gt_image(gt_path)
        class_map = rgb_to_class_map(rgb)
        grid64 = convert_to_grid64(class_map, grid_size=64, weed_threshold=0.05)

        parcel_tag = f"parcel_{idx:03d}_{gt_path.stem}"
        npy_path = out_dir / f"{parcel_tag}_64x64.npy"
        png_path = out_dir / f"{parcel_tag}_64x64.png"

        np.save(npy_path, grid64)
        save_preview(grid64, png_path)

        unique, counts = np.unique(grid64, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(
            f"Saved {npy_path} | shape={grid64.shape} dtype={grid64.dtype} "
            f"| counts(0:soil,1:crop,2:weed)={dist}"
        )


if __name__ == "__main__":
    main()