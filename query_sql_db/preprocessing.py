import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class ImagePreprocessor:
    def __init__(self, image_path: str):
        """Initialize the preprocessor with an image file path."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
    
    def to_grayscale(self):
        """Convert image to grayscale."""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image
    
    def remove_noise(self):
        """Apply Gaussian blur to reduce noise."""
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        return self.image
    
    def thresholding(self):
        """Apply adaptive thresholding for better OCR results."""
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        return self.image
    
    def extract_text(self):
        """Extract text from the processed image using Tesseract OCR."""
        return pytesseract.image_to_string(self.image)
    
    def preprocess_and_extract(self):
        """Run all preprocessing steps and extract text."""
        self.to_grayscale()
        self.remove_noise()
        self.thresholding()
        
        return self.extract_text()