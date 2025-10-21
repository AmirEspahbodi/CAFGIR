import shutil
from pathlib import Path
import sys
import os

# --- Configuration ---
# Set the base paths for your two folders
SOURCE_BASE_DIR = Path("D:\\amir_es\\car_accessories_dataset_augmented\\new_noisy_aug_512")
TARGET_BASE_DIR = Path("D:\\amir_es\\car_accessories_dataset_augmented\\new_aug")
# ---------------------

def main():
    """
    Moves files from source subdirectories to identically named
    target subdirectories.

    If a file name conflict occurs:
    1. It attempts to rename the file by adding "_noise" (e.g., "img.jpg" -> "img_noise.jpg").
    2. If "img_noise.jpg" also exists, it skips the file.
    """
    print(f"🚀 Starting image combination process...")
    print(f"Source folder: {SOURCE_BASE_DIR.resolve()}")
    print(f"Target folder: {TARGET_BASE_DIR.resolve()}")
    print("-" * 40)

    # --- 1. Basic Validation ---
    if not SOURCE_BASE_DIR.is_dir():
        print(f"❌ ERROR: Source directory not found:\n{SOURCE_BASE_DIR}", file=sys.stderr)
        sys.exit(1)  # Exit the script with an error code

    if not TARGET_BASE_DIR.is_dir():
        print(f"❌ ERROR: Target directory not found:\n{TARGET_BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    total_files_moved = 0
    total_files_skipped = 0
    total_classes_processed = 0

    # --- 2. Iterate through source class folders ---
    for source_class_dir in SOURCE_BASE_DIR.iterdir():
        if source_class_dir.is_dir():
            class_name = source_class_dir.name
            print(f"Processing class: {class_name}")
            total_classes_processed += 1

            # --- 3. Determine and prepare target class folder ---
            target_class_dir = TARGET_BASE_DIR / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            files_moved_in_class = 0
            files_skipped_in_class = 0

            # --- 4. Iterate through all files in the source class folder ---
            for source_file in source_class_dir.iterdir():
                if source_file.is_file():
                    target_file = target_class_dir / source_file.name

                    # --- 5. Handle Conflicts (NEW LOGIC) ---
                    if target_file.exists():
                        # --- 5a. Conflict detected! Create a new name ---
                        new_name = f"{source_file.stem}_noise{source_file.suffix}"
                        new_target_file = target_class_dir / new_name

                        # --- 5b. Check if the *new* name also exists ---
                        if new_target_file.exists():
                            print(f"  [!] WARNING: Skipping {source_file.name}. Both {target_file.name} and {new_name} already exist.")
                            files_skipped_in_class += 1
                        else:
                            # --- 5c. New name is available, move and rename ---
                            try:
                                shutil.move(str(source_file), str(new_target_file))
                                print(f"  [i] INFO: Moved and renamed: {source_file.name} -> {new_name}")
                                files_moved_in_class += 1
                            except Exception as e:
                                print(f"  [!] ERROR: Could not move/rename {source_file.name}. Error: {e}")
                                files_skipped_in_class += 1 # Treat a failed move as a skip
                    else:
                        # --- 6. No conflict, just move the file ---
                        try:
                            shutil.move(str(source_file), str(target_file))
                            files_moved_in_class += 1
                        except Exception as e:
                            print(f"  [!] ERROR: Could not move {source_file.name}. Error: {e}")
                            files_skipped_in_class += 1 # Treat a failed move as a skip

            print(f"  > Moved: {files_moved_in_class} files")
            if files_skipped_in_class > 0:
                print(f"  > Skipped: {files_skipped_in_class} files (conflicts)")

            total_files_moved += files_moved_in_class
            total_files_skipped += files_skipped_in_class

            # --- 7. Attempt to clean up empty source directory ---
            if files_skipped_in_class == 0:
                try:
                    source_class_dir.rmdir()
                    print(f"  > Cleaned up empty source directory: {source_class_dir.name}")
                except OSError:
                    print(f"  [!] INFO: Could not remove {source_class_dir.name}. (May not be empty)")
            else:
                print(f"  > Source directory {source_class_dir.name} not removed as {files_skipped_in_class} file(s) were skipped.")
            
            print("-" * 20) # Separator for classes

    print("✅ Process Finished.")
    print("\n--- Summary ---")
    print(f"Total classes processed: {total_classes_processed}")
    print(f"Total files moved: {total_files_moved}")
    print(f"Total files skipped (conflicts): {total_files_skipped}")

if __name__ == "__main__":
    main()