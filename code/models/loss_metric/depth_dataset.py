import os
import numpy as np
import torch
import random
import torch.distributed as dist
import torch.nn.functional as F
import cv2
import imageio
from torch.utils.data import Dataset
from tqdm import tqdm

_METADATA_CACHE = {}

# Base classes for unified dataset structure
class Item:
    """Base class for dataset items"""

    def __init__(self):
        pass

    def unpack(self):
        """Unpack and return the data for this item"""
        raise NotImplementedError("Subclasses must implement unpack method")

    def get_metadata(self):
        """Return metadata about this item (optional)"""
        return {}





class RootDataset(Dataset):
    """Base dataset class that manages a list of Item objects"""

    def __init__(self):
        self.items = []

    def add_items(self, items):
        """Add a list of items to the dataset"""
        self.items.extend(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        max_retries = 10  # Maximum number of retries for corrupted frames
        retries = 0

        while retries < max_retries:
            try:
                # Try to get the item at the current index
                current_idx = (idx + retries) % len(self.items)
                result = self.items[current_idx].unpack()

                # If unpack returns None, it means the frame is corrupted
                if result is None:
                    retries += 1
                    continue

                return result

            except Exception as e:
                print(f"Error loading item at index {current_idx}: {e}")

                retries += 1

                continue

        # If we've exhausted all retries, raise an error
        raise RuntimeError(
            f"Failed to load any valid item after {max_retries} retries starting from index {idx}"
        )


class DepthCarlaHexItem(Item):
    def __init__(self, scene_path, frame_name, keys, args, size=(464, 720), lazy=True):
        super().__init__()
        self.scene_path = scene_path
        self.frame_name = frame_name
        self.keys = keys
        self.args = args
        self.size = size  # (height, width)
        self.lazy = lazy

        # OPTIMIZATION: Lazy initialization - defer heavy I/O until unpack()
        # Saves ~100-500ms per item during dataset creation from cache
        self.calibration = None
        self.original_size = None

        if not lazy:
            # Legacy behavior: load immediately (needed for validation)
            self._load_metadata()

    def _load_metadata(self):
        """Load calibration and original size (called on-demand)"""
        if self.calibration is not None:
            return  # Already loaded

        cache_key = (self.scene_path, tuple(self.keys))
        cached = _METADATA_CACHE.get(cache_key)
        if cached is not None:
            self.calibration, self.original_size = cached
            return

        # np.load(...npz) keeps the zip file handle open until the NpzFile is
        # closed. Copy arrays into memory so long-running workers don't leak
        # descriptors on mounted datasets.
        with np.load(f"{self.scene_path}/calibration.npz") as calibration:
            self.calibration = {key: calibration[key].copy() for key in calibration.files}

        # Get original image size from first available view for intrinsic scaling
        first_key = self.keys[0]
        test_path = f"{self.scene_path}/hdr_{first_key}/{self.frame_name}"
        if not os.path.exists(test_path):
            test_path = test_path.replace(".exr", ".hdr")
        test_img = cv2.imread(test_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if test_img is not None:
            self.original_size = test_img.shape[:2]  # (orig_h, orig_w)
        else:
            # Fallback: assume standard CARLA resolution
            self.original_size = (600, 800)  # Default CARLA camera resolution

        _METADATA_CACHE[cache_key] = (self.calibration, self.original_size)

    def validate_data_quality(self):
        """
        Validate data quality before including in training.
        Returns (is_valid, reason) tuple.

        Optimized version: Fast depth distribution check only.
        """
        try:
            # Sample check: use rear camera (reference view)
            key = "rear" if "rear" in self.keys else self.keys[0]

            # Load depth (use cache if available)
            depth_path = f"{self.scene_path}/ground_truth_depth_{key}/{self.frame_name.replace('.hdr', '.npy').replace('.exr', '.npy')}"
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
            else:
                depth_path = f"{self.scene_path}/ground_truth_depth_{key}/{self.frame_name.replace('.hdr', '.png').replace('.exr', '.png')}"
                if not os.path.exists(depth_path):
                    return False, "depth_not_found"
                depth = (
                    cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    / 1000.0
                )

            # OPTIMIZATION: Subsample for faster validation (use every 4th pixel)
            # This reduces computation by 16x with minimal accuracy loss
            depth_sub = depth[::4, ::4]

            # Check depth distribution
            valid_depth = depth_sub[(depth_sub > 0.1) & (depth_sub < 100.0)]
            if len(valid_depth) < 25:  # Too few valid pixels (adjusted for subsampling)
                return False, "insufficient_valid_depth"

            # Compute depth distribution in bins (fast histogram)
            bins = [0, 5, 15, 30, 50, 100]  # meters
            hist, _ = np.histogram(valid_depth, bins=bins)
            total_valid = len(valid_depth)

            # VERY RELAXED THRESHOLDS: Only reject pathological cases
            near_ratio = hist[0] / total_valid
            if near_ratio > 0.95:
                return False, f"depth_too_near_heavy (near_ratio={near_ratio:.2f})"

            far_ratio = hist[-1] / total_valid
            if far_ratio > 0.95:
                return False, f"depth_too_far_heavy (far_ratio={far_ratio:.2f})"

            mid_ratio = (hist[1] + hist[2] + hist[3]) / total_valid
            if mid_ratio < 0.02:
                return False, f"depth_lacks_mid_range (mid_ratio={mid_ratio:.2f})"

            return True, "valid"

        except Exception as e:
            return False, f"validation_error: {str(e)}"

    def unpack(self):
        # LAZY LOADING: Load metadata on first unpack call
        if self.calibration is None:
            self._load_metadata()

        output = {}
        h, w = self.size

        # Load all available views
        for i, key in enumerate(self.keys):
            # Load HDR image
            frame_path = f"{self.scene_path}/hdr_{key}/{self.frame_name}"
            if not os.path.exists(frame_path):
                frame_path = frame_path.replace(".exr", ".hdr")

            # Fast HDR read
            hdr_np = None
            if frame_path.endswith((".hdr", ".exr")):
                hdr_np = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                if hdr_np is not None:
                    hdr_np = cv2.cvtColor(hdr_np, cv2.COLOR_BGR2RGB)
            if hdr_np is None:
                hdr_np = imageio.imread(frame_path)
            if np.issubdtype(hdr_np.dtype, np.floating):
                hdr_np = np.nan_to_num(hdr_np, nan=0.0, posinf=0.0, neginf=0.0)
            hdr_np = cv2.resize(
                hdr_np.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
            )
            hdr = torch.from_numpy(hdr_np).permute(2, 0, 1).float()

            # Load depth
            depth_path = f"{self.scene_path}/ground_truth_depth_{key}/{self.frame_name.replace('.hdr', '.npy').replace('.exr', '.npy')}"
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
            else:
                depth_path = f"{self.scene_path}/ground_truth_depth_{key}/{self.frame_name.replace('.hdr', '.png').replace('.exr', '.png')}"
                depth = (
                    cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    / 1000.0
                )

            depth = cv2.resize(
                depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
            depth = torch.from_numpy(depth).unsqueeze(0).float()

            # Load camera intrinsic and scale to target resolution
            K_orig = torch.from_numpy(self.calibration[f"K_{key}"]).float()

            # Scale intrinsic matrix for resized image
            # K = [[fx, 0, cx],
            #      [0, fy, cy],
            #      [0,  0,  1]]
            # When resizing from (orig_h, orig_w) to (h, w):
            #   fx_new = fx * (w / orig_w)
            #   fy_new = fy * (h / orig_h)
            #   cx_new = cx * (w / orig_w)
            #   cy_new = cy * (h / orig_h)
            orig_h, orig_w = self.original_size
            scale_x = w / orig_w
            scale_y = h / orig_h

            K = K_orig.clone()
            K[0, 0] *= scale_x  # fx
            K[1, 1] *= scale_y  # fy
            K[0, 2] *= scale_x  # cx
            K[1, 2] *= scale_y  # cy

            # Load camera extrinsic (transform to base frame)
            if key != "rear":  # Assuming 'rear' is the reference frame
                T = torch.from_numpy(self.calibration[f"Transform_{key}_rear"]).float()
            else:
                T = torch.eye(4).float()

            output[f"rgb_{i}"] = hdr
            output[f"depth_{i}"] = depth
            output[f"K_{i}"] = K
            output[f"T_{i}"] = torch.linalg.inv(T).contiguous()  # Pre-compute T_inv

        output["num_views"] = len(self.keys)
        return output


class DepthCarlaHexDataset(RootDataset):
    def __init__(
        self,
        args,
        unified_mode=False,
        img_type="hdr",
        size=(464, 720),
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.args = args
        self.unified_mode = unified_mode
        self.image_type = img_type
        self.size = size
        self.rank = rank
        self.world_size = world_size

        # Allow path override via env var (e.g. for machines where /bean is not mounted)
        calra_root = os.environ.get("CALRA_ROOT", "/bean/calra2")
        excluded_scene_names = {
            name.strip()
            for name in os.environ.get("CALRA_EXCLUDE_SCENES", "").split(",")
            if name.strip()
        }
        included_scene_names = {
            name.strip()
            for name in os.environ.get("CALRA_SCENES", "").split(",")
            if name.strip()
        }
        scenes = os.listdir(calra_root)
        scenes = [
            os.path.join(calra_root, x)
            for x in scenes
            if "hex_type1" in x
            and x not in excluded_scene_names
            and (not included_scene_names or x in included_scene_names)
        ]
        scenes.sort()
        if included_scene_names:
            print(f"[CALRA] Including scenes: {sorted(included_scene_names)}")
        if excluded_scene_names:
            print(f"[CALRA] Excluding scenes: {sorted(excluded_scene_names)}")
        print(scenes)
        # OPTIMIZATION: Only rank 0 does validation, others wait
        if rank == 0:
            print(f"[Rank 0] Starting dataset validation...")
            frames = []
            for scene in scenes:
                frames.extend(self.retrieve_frames(scene))
            #frames = random.sample(frames, len(frames))

            # Save validated frame info for other ranks
            cache_file = "/tmp/validated_frames_cache.pkl"
            import pickle

            frame_info = [(f.scene_path, f.frame_name, f.keys) for f in frames]
            with open(cache_file, "wb") as f:
                pickle.dump(frame_info, f)
            print(f"[Rank 0] Validation complete. {len(frames)} frames validated.")

        # Synchronize all processes
        if world_size > 1:
            dist.barrier()

        # Other ranks load validated results
        if rank != 0:
            import pickle

            cache_file = "/tmp/validated_frames_cache.pkl"
            with open(cache_file, "rb") as f:
                frame_info = pickle.load(f)

            # Reconstruct items from validated info
            frames = [
                DepthCarlaHexItem(scene_path, frame_name, keys, self.args, self.size)
                for scene_path, frame_name, keys in frame_info
            ]
            print(f"[Rank {rank}] Loaded {len(frames)} validated frames from cache.")

        self.add_items(frames)

    def retrieve_frames(self, scene_path):
        # Get available camera keys
        keys = sorted(
            x.split("hdr_")[-1]
            for x in os.listdir(scene_path)
            if x.startswith("hdr_")
            and os.path.isdir(os.path.join(scene_path, x))
            and os.path.isdir(
                os.path.join(scene_path, f"ground_truth_depth_{x.split('hdr_')[-1]}")
            )
        )
        min_required_views = int(getattr(self.args, "num_views", 6))
        if (
            len(keys) < min_required_views
            or "rear" not in keys
            or not os.path.exists(f"{scene_path}/hdr_rear")
        ):
            print(
                f"[Scene {os.path.basename(scene_path)}] Skipping: "
                f"only {len(keys)} complete camera dirs, need >= {min_required_views}"
            )
            return []

        # Get frame names
        frame_names = [
            x
            for x in os.listdir(f"{scene_path}/hdr_rear")
            if x.endswith(f".{self.image_type}")
        ]
        frame_names.sort()

        # OPTIMIZATION: Check persistent validation cache (per scene)
        cache_path = f"{scene_path}/.validation_cache.npz"
        validation_cache = {}
        cache_loaded = False

        use_validation_cache = os.environ.get("CALRA_SKIP_VALIDATION_CACHE", "0") != "1"

        if use_validation_cache and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=True) as cache_data:
                    cache_keys = []
                    if "camera_keys" in cache_data.files:
                        cache_keys = [str(x) for x in cache_data["camera_keys"].tolist()]
                    if cache_keys != keys:
                        raise ValueError(
                            f"camera key mismatch cache={cache_keys} current={keys}"
                        )
                    validation_cache = cache_data["validation_results"].item()
                cache_loaded = True

                # Check if cache is complete (all frames validated)
                cached_frames = set(validation_cache.keys())
                current_frames = set(frame_names)

                if cached_frames >= current_frames:
                    # Cache is complete! No validation needed
                    print(
                        f"[Scene {os.path.basename(scene_path)}] Using complete cache ({len(validation_cache)} frames)"
                    )

                    # FAST PATH: Create items with lazy=True (NO I/O!)
                    # Metadata will be loaded on-demand during training
                    items = []
                    rejected_stats = {}
                    for frame_name in frame_names:
                        is_valid, reason = validation_cache.get(
                            frame_name, (False, "not_in_cache")
                        )
                        if is_valid:
                            # OPTIMIZATION: lazy=True means instant creation (no disk I/O)
                            items.append(
                                DepthCarlaHexItem(
                                    scene_path,
                                    frame_name,
                                    keys,
                                    self.args,
                                    self.size,
                                    lazy=True,
                                )
                            )
                        else:
                            rejected_stats[reason] = rejected_stats.get(reason, 0) + 1

                    # Print statistics
                    if rejected_stats:
                        total_checked = len(frame_names)
                        total_rejected = sum(rejected_stats.values())
                        total_accepted = len(items)
                        print(
                            f"[Scene {os.path.basename(scene_path)}] "
                            f"Accepted: {total_accepted}/{total_checked}, "
                            f"Rejected: {total_rejected} (from cache)"
                        )

                    return items
                else:
                    # Cache exists but incomplete - need to validate new frames
                    new_frame_count = len(current_frames - cached_frames)
                    print(
                        f"[Scene {os.path.basename(scene_path)}] Cache incomplete. Validating {new_frame_count} new frames..."
                    )
            except Exception as e:
                print(
                    f"[Scene {os.path.basename(scene_path)}] Cache corrupted, rebuilding: {e}"
                )
                validation_cache = {}

        # If we reach here, need to validate some/all frames
        items = []
        rejected_stats = {}  # Track rejection reasons
        new_validations = {}  # Track new validation results

        for frame_name in tqdm(
            frame_names, desc=f"Validating {os.path.basename(scene_path)}"
        ):
            # Check if all required data exists for this frame
            has_all_data = True
            for key in keys:
                hdr_path = f"{scene_path}/hdr_{key}/{frame_name}"
                depth_path = f"{scene_path}/ground_truth_depth_{key}/{frame_name.replace('.hdr', '.npy').replace('.exr', '.npy')}"

                if not os.path.exists(hdr_path) and not os.path.exists(
                    hdr_path.replace(".exr", ".hdr")
                ):
                    has_all_data = False
                    break
                if not os.path.exists(depth_path):
                    depth_path = f"{scene_path}/ground_truth_depth_{key}/{frame_name.replace('.hdr', '.png').replace('.exr', '.png')}"
                    if not os.path.exists(depth_path):
                        has_all_data = False
                        break

            if has_all_data:
                # Check cache first
                if frame_name in validation_cache:
                    is_valid, reason = validation_cache[frame_name]
                else:
                    # Data-quality validation only checks depth, so avoid
                    # loading calibration.npz for every frame during cache
                    # rebuilds. On /cool mounts this prevents long stalls and
                    # descriptor buildup.
                    item = DepthCarlaHexItem(
                        scene_path, frame_name, keys, self.args, self.size, lazy=True
                    )

                    # Validate data quality
                    is_valid, reason = item.validate_data_quality()
                    new_validations[frame_name] = (is_valid, reason)

                if is_valid:
                    # Create item for training (lazy=True for fast initialization)
                    if frame_name not in validation_cache:
                        # Reuse already created item (but convert to lazy for training)
                        item.lazy = True  # Enable lazy mode for training
                        items.append(item)
                    else:
                        # Create new item with lazy loading (FAST: no I/O!)
                        items.append(
                            DepthCarlaHexItem(
                                scene_path,
                                frame_name,
                                keys,
                                self.args,
                                self.size,
                                lazy=True,
                            )
                        )
                else:
                    # Track rejection reason
                    rejected_stats[reason] = rejected_stats.get(reason, 0) + 1

        # Save updated cache (only if we did new validations)
        if new_validations:
            try:
                all_validations = {**validation_cache, **new_validations}
                np.savez(
                    cache_path,
                    validation_results=all_validations,
                    camera_keys=np.array(keys, dtype=object),
                )
                print(
                    f"[Scene {os.path.basename(scene_path)}] Saved {len(new_validations)} new validations to cache"
                )
            except Exception as e:
                print(
                    f"[Scene {os.path.basename(scene_path)}] Failed to save cache: {e}"
                )

        # Print rejection statistics for this scene
        if rejected_stats:
            total_checked = len(frame_names)
            total_rejected = sum(rejected_stats.values())
            total_accepted = len(items)
            print(
                f"[Scene {os.path.basename(scene_path)}] "
                f"Accepted: {total_accepted}/{total_checked}, "
                f"Rejected: {total_rejected}"
            )

        return items


def collate_fn(batch, num_views=6, num_target_views=3):
    """
    Custom collate function with FIXED num_views for DDP compatibility.

    Args:
        batch: List of data items
        num_views: Fixed number of views (default=6, set via args)

    Returns:
        Dictionary with:
        - All tensors padded to [B, num_views, ...]
        - view_mask: [B, num_views] - 1 for real views, 0 for padding
    """
    batch_size = len(batch)

    # Initialize lists for input views and target views
    rgbs_in = []
    depths_in = []
    Ks_in = []
    Kinvs_in = []
    Ts_in = []
    view_masks_in = []  # Track which input views are real vs padded

    rgbs_tgt = []
    depths_tgt = []
    Ks_tgt = []
    Kinvs_tgt = []
    Ts_tgt = []
    view_masks_tgt = []  # Track which target views are real vs padded

    for item in batch:
        available_views = item["num_views"]

        # Randomly select number of input views (2 to num_views)
        num_select = random.randint(5, min(available_views, num_views))
        selected_indices = random.sample(range(available_views), num_select)
        selected_indices_target = random.sample(
            range(available_views), min(num_target_views, available_views)
        )

        # Build input view lists
        item_rgbs_in = []
        item_depths_in = []
        item_Ks_in = []
        item_Kinvs_in = []
        item_Ts_in = []
        item_mask_in = []

        for i in selected_indices:
            K = item[f"K_{i}"]
            item_rgbs_in.append(item[f"rgb_{i}"])
            item_depths_in.append(item[f"depth_{i}"])
            item_Ks_in.append(K)
            item_Kinvs_in.append(torch.linalg.inv(K.float()).contiguous())
            item_Ts_in.append(item[f"T_{i}"])
            item_mask_in.append(1.0)

        # Pad inputs to num_views with zero tensors
        while len(item_rgbs_in) < num_views:
            # Create zero tensors with same shape as existing tensors
            item_rgbs_in.append(torch.zeros_like(item_rgbs_in[0]))
            item_depths_in.append(torch.zeros_like(item_depths_in[0]))
            item_Ks_in.append(item_Ks_in[0].clone())
            item_Kinvs_in.append(item_Kinvs_in[0].clone())
            item_Ts_in.append(item[f"T_{0}"].clone())
            item_mask_in.append(0.0)

        # Build target view lists
        item_rgbs_tgt = []
        item_depths_tgt = []
        item_Ks_tgt = []
        item_Kinvs_tgt = []
        item_Ts_tgt = []
        item_mask_tgt = []

        for i in selected_indices_target:
            K = item[f"K_{i}"]
            item_rgbs_tgt.append(item[f"rgb_{i}"])
            item_depths_tgt.append(item[f"depth_{i}"])
            item_Ks_tgt.append(K)
            item_Kinvs_tgt.append(torch.linalg.inv(K.float()).contiguous())
            item_Ts_tgt.append(item[f"T_{i}"])
            item_mask_tgt.append(1.0)

        # Pad targets to num_target_views
        while len(item_rgbs_tgt) < num_target_views:
            item_rgbs_tgt.append(item_rgbs_tgt[-1].clone())
            item_depths_tgt.append(item_depths_tgt[-1].clone())
            item_Ks_tgt.append(item_Ks_tgt[-1].clone())
            item_Kinvs_tgt.append(item_Kinvs_tgt[-1].clone())
            item_Ts_tgt.append(item_Ts_tgt[-1].clone())
            item_mask_tgt.append(0.0)

        rgbs_in.append(torch.stack(item_rgbs_in))
        depths_in.append(torch.stack(item_depths_in))
        Ks_in.append(torch.stack(item_Ks_in))
        Kinvs_in.append(torch.stack(item_Kinvs_in))
        Ts_in.append(torch.stack(item_Ts_in))
        view_masks_in.append(torch.tensor(item_mask_in))

        # --- Additional analysis: select middle 3 views by exposure ---
        # Compute per-view exposures for the (padded) num_views images, but
        # only consider real views (mask==1) when selecting the middle ones.
        try:
            # stack to [num_views, C, H, W]
            stacked_hdrs = torch.stack(item_rgbs_in)
            # simulate exposures; returns (ldr, exposures, noise_std)
            _, exps, _ = simulate_hdr_intensity(stacked_hdrs)
            # exps: [num_views, 1, 1, 1] -> squeeze to [num_views]
            exps = exps.view(-1)
        except Exception:
            # If simulation fails for any reason, fall back to zeros
            exps = torch.zeros(len(item_rgbs_in), dtype=torch.float32)

        mask_tensor = torch.tensor(item_mask_in, dtype=torch.bool)
        real_indices = torch.nonzero(mask_tensor).view(-1).tolist()

        # Decide which indices to select as the middle ones
        mid_k = 3
        if len(real_indices) >= mid_k:
            # sort real indices by exposure (ascending)
            real_exps = [(i, float(exps[i].item())) for i in real_indices]
            real_exps.sort(key=lambda x: x[1])
            real_sorted = [i for i, _ in real_exps]
            start = (len(real_sorted) - mid_k) // 2
            mid_sel = real_sorted[start : start + mid_k]
        else:
            # If fewer than mid_k real views, take all real and pad by repeating last
            if len(real_indices) == 0:
                # fallback: choose first mid_k indices (likely padded zeros)
                mid_sel = list(range(min(mid_k, len(item_rgbs_in))))
                while len(mid_sel) < mid_k:
                    mid_sel.append(mid_sel[-1])
            else:
                mid_sel = real_indices.copy()
                while len(mid_sel) < mid_k:
                    mid_sel.append(real_indices[-1])

        # Build mid-selected lists (preserve order as selected)
        item_rgbs_mid = [item_rgbs_in[i].clone() for i in mid_sel]
        item_depths_mid = [item_depths_in[i].clone() for i in mid_sel]
        item_Ks_mid = [item_Ks_in[i].clone() for i in mid_sel]
        item_Kinvs_mid = [item_Kinvs_in[i].clone() for i in mid_sel]
        item_Ts_mid = [item_Ts_in[i].clone() for i in mid_sel]
        item_mask_mid = [1.0 if (i in real_indices) else 0.0 for i in mid_sel]

        # append mid-selection stacks for this item
        # these lists will be converted to tensors after the batch loop
        if "_rgbs_in_mid" not in locals():
            _rgbs_in_mid = []
            _depths_in_mid = []
            _Ks_in_mid = []
            _Kinvs_in_mid = []
            _Ts_in_mid = []
            _view_masks_in_mid = []
            _mid_indices = []

        _rgbs_in_mid.append(torch.stack(item_rgbs_mid))
        _depths_in_mid.append(torch.stack(item_depths_mid))
        _Ks_in_mid.append(torch.stack(item_Ks_mid))
        _Kinvs_in_mid.append(torch.stack(item_Kinvs_mid))
        _Ts_in_mid.append(torch.stack(item_Ts_mid))
        _view_masks_in_mid.append(torch.tensor(item_mask_mid))
        _mid_indices.append(torch.tensor(mid_sel, dtype=torch.long))

        rgbs_tgt.append(torch.stack(item_rgbs_tgt))
        depths_tgt.append(torch.stack(item_depths_tgt))
        Ks_tgt.append(torch.stack(item_Ks_tgt))
        Kinvs_tgt.append(torch.stack(item_Kinvs_tgt))
        Ts_tgt.append(torch.stack(item_Ts_tgt))
        view_masks_tgt.append(torch.tensor(item_mask_tgt))

    return {
        "rgbs_in": torch.stack(rgbs_in),  # [B, num_views, 6, H, W]
        "depths_in": torch.stack(depths_in),  # [B, num_views, 1, H, W]
        "Ks_in": torch.stack(Ks_in),  # [B, num_views, 3, 3]
        "Kinvs_in": torch.stack(Kinvs_in),  # [B, num_views, 3, 3]
        "Ts_in": torch.stack(Ts_in),  # [B, num_views, 4, 4]
        "view_mask_in": torch.stack(
            view_masks_in
        ),  # [B, num_views] - 1=real, 0=padding
        "rgbs_tgt": torch.stack(rgbs_tgt),  # [B, num_target_views, 6, H, W]
        "depths_tgt": torch.stack(depths_tgt),  # [B, num_target_views, 1, H, W]
        "Ks_tgt": torch.stack(Ks_tgt),  # [B, num_target_views, 3, 3]
        "Kinvs_tgt": torch.stack(Kinvs_tgt),  # [B, num_target_views, 3, 3]
        "Ts_tgt": torch.stack(Ts_tgt),  # [B, num_target_views, 4, 4]
        "view_mask_tgt": torch.stack(view_masks_tgt),  # [B, num_target_views]
        "num_views": num_views,  # Always fixed
        "num_target_views": num_target_views,
        # Middle-3-by-exposure analysis outputs (optional additional diagnostics)
        "rgbs_in_mid": torch.stack(_rgbs_in_mid) if "_rgbs_in_mid" in locals() else None,
        "depths_in_mid": torch.stack(_depths_in_mid) if "_depths_in_mid" in locals() else None,
        "Ks_in_mid": torch.stack(_Ks_in_mid) if "_Ks_in_mid" in locals() else None,
        "Kinvs_in_mid": torch.stack(_Kinvs_in_mid) if "_Kinvs_in_mid" in locals() else None,
        "Ts_in_mid": torch.stack(_Ts_in_mid) if "_Ts_in_mid" in locals() else None,
        "view_mask_in_mid": torch.stack(_view_masks_in_mid) if "_view_masks_in_mid" in locals() else None,
        "mid_indices": torch.stack(_mid_indices) if "_mid_indices" in locals() else None,
    }



@torch.no_grad()
def simulate_hdr_intensity(
    hdr_img,
    target_intensity=0.03,
    bins=256,
    log_bins=True,
    high_pct=99.99,
    exp=None,
    noise_phase=True,
    noise_std= 0.002
):
    """
    Simulate HDR to LDR conversion with target intensity

    Returns:
        ldr: LDR image [B,C,H,W]
        exposures: exposure values [B,1,1,1]
        noise_std: noise standard deviation added (scalar or None)
    """
    if exp is not None:
        return hdr_img * exp, exp, None

    B, C, H, W = hdr_img.shape
    exposures = []

    # Compute exposure per image by binary-searching the exposure that makes
    # the mean of the clipped LDR (after clamp to [0,1]) equal to target_intensity.
    # This accounts for saturation clipping which affects the observed mean.
    for b in range(B):
        img = hdr_img[b]  # [C,H,W]

        # initial naive guess (may be off due to clipping)
        img_mean = img.mean()
        init_exp = (target_intensity / (img_mean + 1e-12)).item()

        # bounds for binary search
        low = 1e-6
        high = max(1e-6, init_exp * 24.0 + 1e-6)

        # ensure high yields mean >= target (expand if necessary)
        for _ in range(10):
            ldr_tmp = torch.clamp(img * high, 0.0, 1.0)
            mean_tmp = float(ldr_tmp.mean().item())
            if mean_tmp >= target_intensity:
                break
            high *= 2.0

        # binary search iterations (20 iters is plenty)
        exp_val = init_exp
        for _ in range(20):
            mid = 0.5 * (low + high)
            ldr_mid = torch.clamp(img * mid, 0.0, 1.0)
            mean_mid = float(ldr_mid.mean().item())
            if mean_mid > target_intensity:
                # too bright → reduce exposure
                high = mid
            else:
                # too dark → increase exposure
                low = mid
            exp_val = mid

        exposures.append(
            torch.tensor(exp_val, device=hdr_img.device, dtype=hdr_img.dtype)
        )

    exposures = torch.stack(exposures).view(B, 1, 1, 1)
    ldr = torch.clamp(hdr_img * exposures, 0.0, 1.0)

    # REMOVED: Light noise was causing training instability
    # Sensor simulation already provides realistic noise

    if noise_phase:
        noise_std = noise_std * (random.random() ** (1 / 2))
        ldr = ldr + torch.randn_like(ldr) * noise_std

    return ldr.clamp(0, 1), exposures, noise_std
