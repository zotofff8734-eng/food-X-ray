import easyocr
import logging

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Model Cache ---
_reader_cache = {}

def load_ocr_model(languages=['ru', 'en']):
    """
    Loads and caches the EasyOCR Reader model.
    The model is downloaded automatically on the first run.
    """
    lang_key = ','.join(sorted(languages))
    if lang_key not in _reader_cache:
        logger.info(f"Loading EasyOCR model for languages: {languages}")
        try:
            # gpu=False to ensure it runs on CPU if CUDA is not available/configured
            reader = easyocr.Reader(languages, gpu=False)
            _reader_cache[lang_key] = reader
            logger.info("EasyOCR model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading EasyOCR model: {e}")
            return None
    return _reader_cache[lang_key]

def recognize_text_from_image(image_path: str) -> str | None:
    """
    Recognizes text from an image file using EasyOCR.
    
    :param image_path: Path to the image file.
    :return: A single string containing all recognized text, or None if failed.
    """
    reader = load_ocr_model()
    if not reader:
        return None

    try:
        logger.info(f"Recognizing text from image: {image_path}")
        # readtext returns a list of (bounding_box, text, confidence) tuples
        results = reader.readtext(image_path, detail=1)
        
        if not results:
            logger.warning("EasyOCR did not find any text in the image.")
            return ""
            
        # Combine all text fragments into a single string
        recognized_text = " ".join([res[1] for res in results])
        
        logger.info(f"Recognized text (preview): '{recognized_text[:100]}...'")
        return recognized_text

    except Exception as e:
        logger.error(f"An error occurred during text recognition: {e}")
        return None

if __name__ == '__main__':
    print("--- Running OCR (EasyOCR) Test ---")
    
    # This test checks if the model can be loaded.
    # The first time you run this, it will download the necessary models.
    print("Attempting to load EasyOCR model...")
    load_ocr_model()
    
    print("
To test full functionality, you need a sample image file.")
    # Example usage:
    # text = recognize_text_from_image('path/to/your/image.jpg')
    # if text is not None:
    #     print(f"
--- Recognized Text ---
{text}")
