#!/usr/bin/env python3
"""
Standalone script for testing document extraction and verification.
This can be used to test the document extraction functionality without running the full web application.
"""

import os
import sys
import cv2
import argparse
import logging
from ultralytics import YOLO
import easyocr
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_uid_from_text(text_data):
    """Extract Aadhaar UID number from text using regex pattern"""
    # Look for 12-digit number pattern commonly found in Aadhaar
    uid_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    matches = re.findall(uid_pattern, text_data)
    if matches:
        # Clean up the UID by removing spaces
        return re.sub(r'\s+', '', matches[0])
    return None

def extract_name_from_text(text_data):
    """Extract name from document text"""
    name_patterns = [
        r'(?:Name|नाम)[:\s]+([A-Za-z\s]+)',  # Look for "Name:" or "नाम:" followed by text
        r'(?:[T|t]o\s+)([A-Za-z\s]+)'  # Look for "To" followed by a name
    ]
    
    for pattern in name_patterns:
        matches = re.search(pattern, text_data)
        if matches:
            return matches.group(1).strip()
    
    return None

def extract_address_from_text(text_data):
    """Extract address from document text"""
    address_patterns = [
        r'(?:Address|पता)[:\s]+(.+?)(?:\n|$)',  # Look for "Address:" or "पता:" followed by text
        r'(?:[A|a]ddress:?)(.+?)(?:\n\n|\n[A-Z]|$)'  # More general pattern
    ]
    
    for pattern in address_patterns:
        matches = re.search(pattern, text_data, re.DOTALL)
        if matches:
            return matches.group(1).strip()
    
    return None

def classify_doc_from_text(text_data):
    """Classify document type based on text content"""
    text_lower = text_data.lower()
    
    if 'aadhaar' in text_lower or 'unique identification authority' in text_lower or 'आधार' in text_lower:
        return 'Aadhaar'
    elif 'election commission' in text_lower or 'voter id' in text_lower or 'voter identity card' in text_lower:
        return 'Voter ID'
    elif 'passport' in text_lower or ('republic of india' in text_lower and 'passport' in text_lower):
        return 'Passport'
    elif 'driving' in text_lower and 'licence' in text_lower:
        return 'Driving License'
    elif 'pan' in text_lower and ('income tax' in text_lower or 'permanent account' in text_lower):
        return 'PAN Card'
    else:
        return "Unknown"

def extract_text(image_path, reader):
    """Extract text from image using OCR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return []
            
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract text
        result = reader.readtext(gray, detail=0, paragraph=True)
        
        return result
    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {str(e)}")
        return []

def extract_fields_with_detection(image_path, detector, reader):
    """Extract document fields using detection model and OCR"""
    extracted_data = {
        'uid': None,
        'name': None,
        'address': None
    }
    doc_type = "Unknown"
    
    try:
        # Check if file exists
        if not os.path.isfile(image_path):
            logger.error(f"File does not exist: {image_path}")
            return extracted_data, doc_type
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return extracted_data, doc_type
        
        # Run OCR on full image to determine document type
        full_text = ' '.join(extract_text(image_path, reader))
        doc_type = classify_doc_from_text(full_text)
        
        # If detector model is available, use it for field extraction
        if detector:
            results = detector(image_path)
            
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(class_id)
                
                # Get field type
                field_class = results[0].names[class_id].lower()
                
                # Crop region
                try:
                    cropped_roi = image[y1:y2, x1:x2]
                    if cropped_roi.size == 0:
                        continue
                        
                    # Convert to grayscale
                    gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Extract text from region
                    text_results = reader.readtext(gray_roi, detail=0)
                    text = ' '.join(text_results).strip()
                    
                    # Store by field type
                    if field_class == 'name' and text:
                        extracted_data['name'] = text
                    elif field_class == 'uid' and text:
                        # Clean UID (remove spaces, special chars)
                        extracted_data['uid'] = re.sub(r'\D', '', text)
                    elif field_class == 'address' and text:
                        extracted_data['address'] = text
                except Exception as e:
                    logger.error(f"Error processing region ({field_class}): {str(e)}")
        
        # If fields are still missing, try to extract from full text
        if not extracted_data['uid']:
            extracted_data['uid'] = extract_uid_from_text(full_text)
        if not extracted_data['name']:
            extracted_data['name'] = extract_name_from_text(full_text)
        if not extracted_data['address']:
            extracted_data['address'] = extract_address_from_text(full_text)
            
        return extracted_data, doc_type
        
    except Exception as e:
        logger.error(f"Error in field extraction: {str(e)}")
        return extracted_data, "Error"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Document Field Extraction Tool')
    parser.add_argument('image_path', help='Path to the document image')
    parser.add_argument('--detection-model', default='detection_model/runs/detect/train13/weights/best.pt',
                      help='Path to YOLO detection model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        return 1
    
    # Initialize models
    try:
        logger.info("Initializing OCR...")
        reader = easyocr.Reader(['en'])
        
        logger.info(f"Loading detection model from {args.detection_model}...")
        detector = None
        if os.path.exists(args.detection_model):
            detector = YOLO(args.detection_model)
        else:
            logger.warning(f"Detection model not found at {args.detection_model}. Using OCR-only extraction.")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return 1
    
    # Extract fields
    logger.info(f"Processing image: {args.image_path}")
    extracted_data, doc_type = extract_fields_with_detection(args.image_path, detector, reader)
    
    # Print results
    print("\n====== EXTRACTION RESULTS ======")
    print(f"Document Type: {doc_type}")
    print(f"UID: {extracted_data['uid'] or 'Not detected'}")
    print(f"Name: {extracted_data['name'] or 'Not detected'}")
    print(f"Address: {extracted_data['address'] or 'Not detected'}")
    print("================================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())