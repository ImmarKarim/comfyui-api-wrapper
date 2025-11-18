"""
Image Compression Utility for ComfyUI API Wrapper

Provides JPEG compression to reduce file sizes while maintaining visual quality.
Also strips metadata for security (prevents reverse-engineering of ComfyUI workflows).

Key Features:
- Converts images to JPEG format at quality 95 (optimal balance)
- Strips all EXIF/metadata to prevent workflow leakage
- Handles transparency by compositing onto white background
- Reduces typical file sizes from ~5MB to ~800KB

Usage:
    from utils.image_compression import compress_image_file
    
    # Compress in-place (replaces original)
    compress_image_file("/path/to/image.png")
    
    # Compress to new location
    compress_image_file("/path/to/image.png", output_path="/path/to/compressed.jpg")
"""

import logging
from pathlib import Path
from typing import Optional
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)

# Optimal JPEG quality for ComfyUI outputs
# 95 provides near-lossless visual quality while achieving ~85% size reduction
DEFAULT_JPEG_QUALITY = 95


def compress_image_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    replace_original: bool = True
) -> Path:
    """
    Compress an image file to JPEG format with metadata stripping.
    
    Args:
        input_path: Path to the input image file
        output_path: Optional path for output. If None, uses input_path with .jpg extension
        jpeg_quality: JPEG quality setting (1-100). Default is 95 for optimal balance
        replace_original: If True and output_path is None, replaces the original file
        
    Returns:
        Path to the compressed output file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If image cannot be processed
        
    Example:
        >>> # Compress PNG to JPEG, replacing original
        >>> compress_image_file("output/image.png")
        PosixPath('output/image.jpg')
        
        >>> # Compress to specific location
        >>> compress_image_file("output/image.png", "compressed/image.jpg")
        PosixPath('compressed/image.jpg')
    """
    input_path = Path(input_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if output_path is None:
        if replace_original:
            # Replace original file (change extension to .jpg)
            output_path = input_path.with_suffix('.jpg')
        else:
            # Create new file with _compressed suffix
            output_path = input_path.with_stem(f"{input_path.stem}_compressed").with_suffix('.jpg')
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the image
        logger.debug(f"Loading image: {input_path}")
        original_image = Image.open(input_path)
        original_size = input_path.stat().st_size
        
        # Convert to JPEG format and strip metadata
        logger.debug(f"Converting to JPEG (quality={jpeg_quality}), mode={original_image.mode}, size={original_image.size}")
        compressed_image = _convert_to_jpeg(original_image, jpeg_quality)
        
        # Save compressed image
        compressed_image.save(output_path, 'JPEG', quality=jpeg_quality, optimize=True)
        compressed_size = output_path.stat().st_size
        
        # Calculate compression ratio
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(
            f"Compressed {input_path.name}: "
            f"{original_size / 1024:.1f}KB → {compressed_size / 1024:.1f}KB "
            f"({compression_ratio:.1f}% reduction)"
        )
        
        # If we're replacing the original and extensions differ, remove the original
        if replace_original and input_path != output_path and input_path.exists():
            logger.debug(f"Removing original file: {input_path}")
            input_path.unlink()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing image {input_path}: {e}")
        raise ValueError(f"Failed to compress image: {str(e)}") from e


def _convert_to_jpeg(pil_image: Image.Image, quality: int) -> Image.Image:
    """
    Convert PIL image to JPEG format and strip all metadata.
    
    This function:
    1. Converts RGBA/LA/P modes to RGB (JPEG doesn't support transparency)
    2. Composites transparent images onto white background
    3. Strips all EXIF/metadata by saving and reloading
    
    Args:
        pil_image: Source PIL Image
        quality: JPEG quality (1-100)
        
    Returns:
        Clean PIL Image in RGB mode with no metadata
    """
    # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        # Create white background
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        
        # Convert palette mode to RGBA first
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        
        # Paste image onto white background, using alpha channel as mask
        if pil_image.mode in ('RGBA', 'LA'):
            background.paste(pil_image, mask=pil_image.split()[-1])
        else:
            background.paste(pil_image)
        
        pil_image = background
        
    elif pil_image.mode not in ('RGB', 'L'):
        # Convert other modes to RGB
        pil_image = pil_image.convert('RGB')
    
    # Strip metadata by saving to memory buffer and reloading
    # This ensures no EXIF data, ComfyUI metadata, or other info is preserved
    temp_buffer = BytesIO()
    pil_image.save(temp_buffer, format='JPEG', quality=quality, optimize=True)
    temp_buffer.seek(0)
    
    # Load clean image (no metadata)
    clean_image = Image.open(temp_buffer)
    clean_image.load()  # Ensure image data is fully loaded into memory
    
    return clean_image


async def compress_image_file_async(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    replace_original: bool = True
) -> Path:
    """
    Async wrapper for compress_image_file.
    
    Runs compression in an executor to avoid blocking the event loop.
    Use this in async contexts (like FastAPI endpoints or async workers).
    
    Args:
        Same as compress_image_file()
        
    Returns:
        Path to the compressed output file
        
    Example:
        >>> await compress_image_file_async("output/image.png")
        PosixPath('output/image.jpg')
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        compress_image_file,
        input_path,
        output_path,
        jpeg_quality,
        replace_original
    )

