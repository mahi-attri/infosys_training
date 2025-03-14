import os
import zipfile
import pandas as pd
import cv2
from ultralytics import YOLO
import easyocr
import re
import difflib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import traceback
import uuid
import logging
import shutil
import time
import threading
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, Response
from flask_cors import CORS
from io import BytesIO
from datetime import datetime

# Initialize database connection variable
DB_CONNECTION = None

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and directory setup
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
VISUALIZATIONS_DIR = os.path.join(STATIC_DIR, 'visualizations')
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Add function to standardize document types
def standardize_document_type(doc_type):
    """
    Standardize document type name formatting
    
    Args:
        doc_type: Original document type string
    
    Returns:
        standardized_doc_type: Formatted document type string
    """
    if not doc_type:
        return "Unknown"
        
    # Convert to string and lowercase for comparison
    doc_type_lower = str(doc_type).lower().replace('_', ' ')
    
    # Standardize document types
    if doc_type_lower in ['aadhaar', 'aadhar']:
        return 'Aadhaar'
    elif doc_type_lower in ['non aadhaar', 'non aadhar', 'non-aadhaar', 'non-aadhar', 'non_aadhaar', 'non_aadhar']:
        return 'Non-Aadhaar'  # Ensure consistent capitalization
    elif doc_type_lower in ['voter id', 'voterid', 'voter_id']:
        return 'Voter ID'
    elif doc_type_lower in ['driving license', 'drivinglicense', 'driving_license']:
        return 'Driving License'
    elif doc_type_lower in ['pan card', 'pancard', 'pan_card']:
        return 'PAN Card'
    elif doc_type_lower == 'passport':
        return 'Passport'
    else:
        # Capitalize each word for other document types
        return ' '.join(word.capitalize() for word in doc_type_lower.split())

# Initialize models
# Update this section in your code to correct the model paths
def initialize_models():
    """Initialize all required models for document processing"""
    models = {}
    
    # Initialize YOLO detection model - use absolute path without MODELS_DIR
    try:
        detection_model_path = "uploads/models/detection.pt"  # Direct path without using os.path.join with MODELS_DIR
        if os.path.exists(detection_model_path):
            models['detector'] = YOLO(detection_model_path)
            logger.info(f"Detection model loaded from {detection_model_path}")
        else:
            logger.warning(f"Detection model file not found at {detection_model_path}")
            models['detector'] = None
    except Exception as e:
        logger.error(f"Error loading detection model: {str(e)}")
        models['detector'] = None
    
    # Initialize YOLO classification model - use absolute path without MODELS_DIR
    try:
        classification_model_path = "uploads/models/classification.pt"  # Direct path without using os.path.join with MODELS_DIR
        if os.path.exists(classification_model_path):
            models['classifier'] = YOLO(classification_model_path)
            logger.info(f"Classification model loaded from {classification_model_path}")
        else:
            logger.warning(f"Classification model file not found at {classification_model_path}")
            models['classifier'] = None
    except Exception as e:
        logger.error(f"Error loading classification model: {str(e)}")
        models['classifier'] = None
    
    # Initialize EasyOCR
    try:
        models['reader'] = easyocr.Reader(['en'])
        logger.info("EasyOCR initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing EasyOCR: {str(e)}")
        models['reader'] = None
    
    return models

# Flask application setup
app = Flask(__name__, static_folder=STATIC_DIR, template_folder='templates')
app.secret_key = os.getenv("SECRET_KEY", "aadharverify_secret_key")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
CORS(app)

# Initialize the models at startup
MODELS = initialize_models()

# Initialize database connection
try:
    import mysql_connection as db
    DB_CONNECTION = db.create_connection()
    if DB_CONNECTION:
        logger.info("Successfully connected to MySQL database")
    else:
        logger.warning("Could not connect to MySQL database, proceeding without database storage")
except Exception as e:
    logger.error(f"Error initializing database connection: {str(e)}")
    DB_CONNECTION = None

# Routes
@app.route('/', endpoint='root')  # Added unique endpoint name
@app.route('/home', endpoint='home')  # Added unique endpoint name
def index():
    """Render the home page"""
    return render_template('index.html')

# Document verification functions
def is_valid_image(file_path):
    """Check if the file is a valid image that can be processed"""
    try:
        if not os.path.isfile(file_path):
            logger.warning(f"Not a file: {file_path}")
            return False
            
        img = Image.open(file_path)
        img.verify()  # Verify it's an image
        return True
    except Exception as e:
        logger.warning(f"Invalid image file {file_path}: {str(e)}")
        return False

def extract_text_with_detection(image_path, models):
    """
    Extract text from document using YOLO detection model and EasyOCR
    
    Args:
        image_path: Path to the image file
        models: Dictionary containing initialized models
    
    Returns:
        extracted_data: Dictionary with detected fields (name, uid, address)
        doc_type: Type of document detected
    """
    extracted_data = {
        'uid': None,
        'name': None,
        'address': None
    }
    
    try:
        if not is_valid_image(image_path):
            return extracted_data, "Invalid"

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return extracted_data, "Invalid"
            
        # Detect document type first using classifier model
        doc_type = "Unknown"
        
        if models['classifier']:
            try:
                class_results = models['classifier'](image_path)
                # Get class with highest confidence
                classes = class_results[0].probs.top5
                class_names = [class_results[0].names[idx] for idx in classes]
                
                # Set document type from classification result
                doc_type = class_names[0]  # Highest confidence class
                # Standardize the document type
                doc_type = standardize_document_type(doc_type)
            except Exception as e:
                logger.error(f"Error during document classification: {str(e)}")
                # Fall back to OCR-based classification
                doc_type = classify_doc_from_text(image, models['reader'])
        else:
            # If no classifier model is available, use OCR-based classification
            doc_type = classify_doc_from_text(image, models['reader'])
            
        # If detection model is available, use it to extract information
        if models['detector']:
            # Run detection
            results = models['detector'](image_path)
            
            # Process detection results
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(class_id)
                
                # Get field type
                field_class = results[0].names[class_id].lower()
                
                # Crop the detected region
                cropped_roi = image[y1:y2, x1:x2]
                
                # Skip processing if the crop is empty
                if cropped_roi.size == 0:
                    continue
                
                # Convert to grayscale for better OCR
                gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
                
                # Use EasyOCR to extract text
                if models['reader']:
                    text_results = models['reader'].readtext(gray_roi, detail=0)
                    extracted_text = ' '.join(text_results).strip()
                    
                    # Store extracted text by field type
                    if field_class == 'name' and not extracted_data['name']:
                        extracted_data['name'] = extracted_text
                    elif field_class == 'uid' and not extracted_data['uid']:
                        # Clean UID (remove spaces, special chars)
                        extracted_data['uid'] = re.sub(r'\D', '', extracted_text)
                    elif field_class == 'address' and not extracted_data['address']:
                        extracted_data['address'] = extracted_text
        else:
            # Fallback to traditional OCR approach if no detector model
            if models['reader']:
                # Process the entire image with OCR
                text_results = models['reader'].readtext(image, detail=0, paragraph=True)
                full_text = ' '.join(text_results)
                
                # Extract UID, name and address using regex patterns
                extracted_data['uid'] = extract_uid_from_text(full_text)
                extracted_data['name'] = extract_name_from_text(full_text)
                extracted_data['address'] = extract_address_from_text(full_text)
                
        return extracted_data, doc_type
        
    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {str(e)}")
        return extracted_data, "Error"

# Update the document classification function to properly identify Aadhaar documents
def classify_doc_from_text(image, reader):
    """Classify document type using OCR text content with improved filtering"""
    if not reader:
        return "Unknown"
        
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(gray, detail=0, paragraph=True)
        text_data = ' '.join(result).lower()
        
        # First check for Voter ID / Election Commission
        if 'election commission' in text_data or 'voter id' in text_data or 'voter identity card' in text_data:
            return 'Voter ID'
            
        # Then check for Aadhaar
        if 'aadhaar' in text_data or 'aadhar' in text_data or 'आधार' in text_data or 'unique identification' in text_data:
            return 'Aadhaar'
            
        # Then other document types
        elif 'passport' in text_data or ('republic of india' in text_data and 'passport' in text_data):
            return 'Passport'
        elif 'driving' in text_data and 'licence' in text_data:
            return 'Driving License'
        elif 'pan' in text_data and ('income tax' in text_data or 'permanent account' in text_data):
            return 'PAN Card'
        else:
            return "Unknown"
    except Exception as e:
        logger.error(f"Error in document classification: {str(e)}")
        return "Unknown"
    
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
    if not text_data:
        return None
    
    # Skip common false-positive organization names
    if any(term in text_data.lower() for term in ['election commission', 'government of india']):
        # For voter ID cards, try to find the actual name after these terms
        lines = text_data.split('\n')
        for i, line in enumerate(lines):
            if 'election commission' in line.lower() and i + 1 < len(lines):
                # Check the next few lines for potential names
                for j in range(1, min(4, len(lines) - i)):
                    potential_name = lines[i + j].strip()
                    # Name pattern: 2-3 words, each word starting with uppercase, no digits
                    if (1 < len(potential_name.split()) < 4 and 
                        all(word[0].isupper() for word in potential_name.split() if word) and
                        not any(char.isdigit() for char in potential_name)):
                        return potential_name
    
    # Normal name patterns
    name_patterns = [
        r'(?:Name|नाम)[:\s]+([A-Za-z\s]+)',  # Look for "Name:" or "नाम:" followed by text
        r'(?:[T|t]o\s+)([A-Za-z\s]+)'  # Look for "To" followed by a name
    ]
    
    for pattern in name_patterns:
        matches = re.search(pattern, text_data)
        if matches:
            name = matches.group(1).strip()
            # Filter out common false positives
            if any(term in name.lower() for term in ['election', 'commission', 'government']):
                continue
            return name
    
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

def reconstruct_address(row):
    """Combine address components into a full address"""
    address_parts = []
    
    # Check various possible column names for address components
    possible_fields = [
        ('House Flat Number', 'House', 'Flat', 'Building', 'HouseNumber'),
        ('Street Road Name', 'Street', 'Road', 'Lane', 'StreetName'),
        ('City', 'Town', 'Village'),
        ('State', 'Province', 'Region'),
        ('PINCODE', 'Zip', 'ZipCode', 'PostalCode', 'PIN')
    ]
    
    for field_options in possible_fields:
        for field in field_options:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                address_parts.append(str(row[field]).strip())
                break
    
    # If Full Address is already provided, use that instead
    if 'Full Address' in row and pd.notna(row['Full Address']) and str(row['Full Address']).strip():
        return str(row['Full Address']).strip()
    
    # If Address is provided, use that
    if 'Address' in row and pd.notna(row['Address']) and str(row['Address']).strip():
        return str(row['Address']).strip()
                
    return ', '.join(address_parts)

def name_match(input_name, extracted_name):
    """
    Calculate if names match based on specified matching rules.
    Returns True if the names match according to defined rules, False otherwise.
    """
    if not input_name or not extracted_name:
        return False
    
    # Normalize names - lowercase and remove extra spaces
    input_normalized = ' '.join(input_name.lower().split())
    extracted_normalized = ' '.join(extracted_name.lower().split())
    
    # Rule 1: Exact Match
    if input_normalized == extracted_normalized:
        return True
    
    # Convert names to lists of parts
    input_parts = input_normalized.split()
    extracted_parts = extracted_normalized.split()
    
    # Rule 2: Abbreviated Names
    # Check if first name is abbreviated (e.g., "J Smith" vs "John Smith")
    if len(input_parts) > 0 and len(extracted_parts) > 0:
        # Check first name abbreviation
        if len(input_parts[0]) == 1 and extracted_parts[0].startswith(input_parts[0]):
            # Compare rest of the name
            if ' '.join(input_parts[1:]) == ' '.join(extracted_parts[1:]):
                return True
                
        # Check if extracted name has abbreviation
        if len(extracted_parts[0]) == 1 and input_parts[0].startswith(extracted_parts[0]):
            # Compare rest of the name
            if ' '.join(input_parts[1:]) == ' '.join(extracted_parts[1:]):
                return True
    
    # Rule 3: Ignoring Middle Names
    if len(input_parts) >= 2 and len(extracted_parts) >= 2:
        # Check if first and last names match (ignoring middle names)
        if input_parts[0] == extracted_parts[0] and input_parts[-1] == extracted_parts[-1]:
            return True
    
    # Rule 4: Matching Any Part of the Name
    # Check if one name is a subset of the other
    if len(input_parts) == 1 and input_parts[0] in extracted_parts:
        return True
    if len(extracted_parts) == 1 and extracted_parts[0] in input_parts:
        return True
    
    # Rule 5: Circular Matching - Check if all parts are present regardless of order
    if set(input_parts) == set(extracted_parts):
        return True
    
    # Rule 6: Single-Letter Abbreviation - in any part of the name
    # Check if one name has a single letter that could be an abbreviation of a part in the other name
    for i, part1 in enumerate(input_parts):
        if len(part1) == 1:
            # Check if this could be an abbreviation of any part in extracted_parts
            for part2 in extracted_parts:
                if part2.startswith(part1):
                    # Create copies of the name parts without the current parts
                    input_copy = input_parts.copy()
                    extracted_copy = extracted_parts.copy()
                    input_copy.pop(i)
                    extracted_copy.remove(part2)
                    # Check if the remaining parts match (regardless of order)
                    if set(input_copy) == set(extracted_copy):
                        return True
    
    # Check the same for extracted_parts
    for i, part1 in enumerate(extracted_parts):
        if len(part1) == 1:
            # Check if this could be an abbreviation of any part in input_parts
            for part2 in input_parts:
                if part2.startswith(part1):
                    # Create copies of the name parts without the current parts
                    input_copy = input_parts.copy()
                    extracted_copy = extracted_parts.copy()
                    extracted_copy.pop(i)
                    input_copy.remove(part2)
                    # Check if the remaining parts match (regardless of order)
                    if set(input_copy) == set(extracted_copy):
                        return True
    
    # If no match found by this point, return False
    return False

def address_match(input_address, extracted_address):
    """
    Calculate if addresses match using defined rules.
    Returns True if the address match score is above the threshold, False otherwise.
    """
    if not input_address or not extracted_address:
        return False
    
    # Normalization - Step 1
    # List of common terms to ignore
    common_terms = ['marg', 'lane', 'township', 'road', 'street', 'rd', 'st', 'avenue', 'ave']
    
    # Normalize addresses
    def normalize_address(addr):
        addr_lower = addr.lower()
        # Remove common terms
        for term in common_terms:
            addr_lower = addr_lower.replace(f' {term} ', ' ')
            # Also check at the end of address
            if addr_lower.endswith(f' {term}'):
                addr_lower = addr_lower[:-len(term)-1]
                
        # Remove non-alphanumeric and normalize spaces
        normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in addr_lower)
        return ' '.join(normalized.split())
    
    normalized_input = normalize_address(input_address)
    normalized_extracted = normalize_address(extracted_address)
    
    # Pincode Matching - Step 2
    # Extract pincodes (6-digit numbers) from addresses
    input_pincode_match = re.search(r'\b(\d{6})\b', input_address)
    extracted_pincode_match = re.search(r'\b(\d{6})\b', extracted_address)
    
    pincode_score = 0
    if input_pincode_match and extracted_pincode_match:
        input_pincode = input_pincode_match.group(1)
        extracted_pincode = extracted_pincode_match.group(1)
        if input_pincode == extracted_pincode:
            pincode_score = 100
    
    # Field-Specific Matching - Step 3
    # Extract individual components from addresses
    input_components = normalized_input.split()
    extracted_components = normalized_extracted.split()
    
    # Calculate component match scores
    component_scores = []
    for comp in input_components:
        if comp in extracted_components:
            component_scores.append(100)  # Exact match
        else:
            # Find best partial match using similarity ratio
            best_score = 0
            for e_comp in extracted_components:
                similarity = difflib.SequenceMatcher(None, comp, e_comp).ratio() * 100
                best_score = max(best_score, similarity)
            component_scores.append(best_score)
    
    # Final Address Match - Step 4
    # Calculate weighted average of scores
    if not component_scores:
        component_avg = 0
    else:
        component_avg = sum(component_scores) / len(component_scores)
    
    # Final score (weighted combination of component match and pincode match)
    component_weight = 0.7
    pincode_weight = 0.3
    final_score = (component_avg * component_weight) + (pincode_score * pincode_weight)
    
    # Return True if score meets threshold
    return final_score >= 70  # 70% threshold

# Add new functions to calculate matching scores
def calculate_uid_match_score(extracted_uid, input_uid):
    """
    Calculate a matching score for UIDs
    
    Args:
        extracted_uid: Extracted UID from document
        input_uid: Input UID from reference data
        
    Returns:
        match_score: Score between 0-100 indicating match quality
    """
    if not extracted_uid or not input_uid:
        return 0
    
    # Clean both UIDs to remove spaces and special characters
    extracted_clean = re.sub(r'\D', '', str(extracted_uid))
    input_clean = re.sub(r'\D', '', str(input_uid))
    
    # For UID, we want an exact match, but we'll calculate partial match for feedback
    if extracted_clean == input_clean:
        return 100
    
    # Find the longest matching substring
    common_length = 0
    for i in range(min(len(extracted_clean), len(input_clean))):
        if extracted_clean[i] == input_clean[i]:
            common_length += 1
    
    # Calculate score based on percentage of matching characters
    if max(len(extracted_clean), len(input_clean)) > 0:
        score = (common_length / max(len(extracted_clean), len(input_clean))) * 100
    else:
        score = 0
        
    return round(score, 2)

def calculate_name_match_score(input_name, extracted_name):
    """
    Calculate a matching score for names
    
    Args:
        input_name: Input name from reference data
        extracted_name: Extracted name from document
        
    Returns:
        match_score: Score between 0-100 indicating match quality
        match_type: Type of match that was found
    """
    if not input_name or not extracted_name:
        return 0, "No match"
    
    # Normalize names - lowercase and remove extra spaces
    input_normalized = ' '.join(input_name.lower().split())
    extracted_normalized = ' '.join(extracted_name.lower().split())
    
    # Rule 1: Exact Match
    if input_normalized == extracted_normalized:
        return 100, "Exact match"
    
    # Convert names to lists of parts
    input_parts = input_normalized.split()
    extracted_parts = extracted_normalized.split()
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, input_normalized, extracted_normalized).ratio() * 100
    
    # Rule 2: Abbreviated Names
    if len(input_parts) > 0 and len(extracted_parts) > 0:
        # Check first name abbreviation
        if len(input_parts[0]) == 1 and extracted_parts[0].startswith(input_parts[0]):
            # Compare rest of the name
            if ' '.join(input_parts[1:]) == ' '.join(extracted_parts[1:]):
                return 90, "Abbreviated name match"
                
        # Check if extracted name has abbreviation
        if len(extracted_parts[0]) == 1 and input_parts[0].startswith(extracted_parts[0]):
            # Compare rest of the name
            if ' '.join(input_parts[1:]) == ' '.join(extracted_parts[1:]):
                return 90, "Abbreviated name match"
    
    # Rule 3: Ignoring Middle Names
    if len(input_parts) >= 2 and len(extracted_parts) >= 2:
        # Check if first and last names match (ignoring middle names)
        if input_parts[0] == extracted_parts[0] and input_parts[-1] == extracted_parts[-1]:
            return 85, "Match ignoring middle name"
    
    # Rule 4: Matching Any Part of the Name
    if len(input_parts) == 1 and input_parts[0] in extracted_parts:
        return 70, "Partial name match"
    if len(extracted_parts) == 1 and extracted_parts[0] in input_parts:
        return 70, "Partial name match"
    
    # Rule 5: Circular Matching - Check if all parts are present regardless of order
    if set(input_parts) == set(extracted_parts):
        return 80, "Name parts match (different order)"
    
    # Rule 6: Single-Letter Abbreviation - in any part of the name
    for i, part1 in enumerate(input_parts):
        if len(part1) == 1:
            # Check if this could be an abbreviation of any part in extracted_parts
            for part2 in extracted_parts:
                if part2.startswith(part1):
                    # Create copies of the name parts without the current parts
                    input_copy = input_parts.copy()
                    extracted_copy = extracted_parts.copy()
                    input_copy.pop(i)
                    extracted_copy.remove(part2)
                    # Check if the remaining parts match (regardless of order)
                    if set(input_copy) == set(extracted_copy):
                        return 75, "Abbreviation match"
    
    # Check the same for extracted_parts
    for i, part1 in enumerate(extracted_parts):
        if len(part1) == 1:
            # Check if this could be an abbreviation of any part in input_parts
            for part2 in input_parts:
                if part2.startswith(part1):
                    # Create copies of the name parts without the current parts
                    input_copy = input_parts.copy()
                    extracted_copy = extracted_parts.copy()
                    extracted_copy.pop(i)
                    input_copy.remove(part2)
                    # Check if the remaining parts match (regardless of order)
                    if set(input_copy) == set(extracted_copy):
                        return 75, "Abbreviation match"
    
    # If no specific rule match is found, return the similarity ratio
    return round(similarity, 2), "Partial similarity"

def calculate_address_match_score(input_address, extracted_address):
    """
    Calculate a matching score for addresses
    
    Args:
        input_address: Input address from reference data
        extracted_address: Extracted address from document
        
    Returns:
        match_score: Score between 0-100 indicating match quality
        match_details: Dictionary with additional match details
    """
    if not input_address or not extracted_address:
        return 0, {"pincode_match": False, "component_match": 0}
    
    # Normalization - Step 1
    # List of common terms to ignore
    common_terms = ['marg', 'lane', 'township', 'road', 'street', 'rd', 'st', 'avenue', 'ave']
    
    # Normalize addresses
    def normalize_address(addr):
        addr_lower = addr.lower()
        # Remove common terms
        for term in common_terms:
            addr_lower = addr_lower.replace(f' {term} ', ' ')
            # Also check at the end of address
            if addr_lower.endswith(f' {term}'):
                addr_lower = addr_lower[:-len(term)-1]
                
        # Remove non-alphanumeric and normalize spaces
        normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in addr_lower)
        return ' '.join(normalized.split())
    
    normalized_input = normalize_address(input_address)
    normalized_extracted = normalize_address(extracted_address)
    
    # Pincode Matching - Step 2
    # Extract pincodes (6-digit numbers) from addresses
    input_pincode_match = re.search(r'\b(\d{6})\b', input_address)
    extracted_pincode_match = re.search(r'\b(\d{6})\b', extracted_address)
    
    # Initialize pincode_score to 0
    pincode_score = 0
    pincode_match = False
    
    if input_pincode_match and extracted_pincode_match:
        input_pincode = input_pincode_match.group(1)
        extracted_pincode = extracted_pincode_match.group(1)
        
        # Check if pincodes match
        if input_pincode == extracted_pincode:
            pincode_match = True
            pincode_score = 100
    
    # Field-Specific Matching - Step 3
    # Extract individual components from addresses
    input_components = normalized_input.split()
    extracted_components = normalized_extracted.split()
    
    # Calculate component match scores
    component_scores = []
    matching_components = []
    for comp in input_components:
        if comp in extracted_components:
            component_scores.append(100)  # Exact match
            matching_components.append(comp)
        else:
            # Find best partial match using similarity ratio
            best_score = 0
            best_match = None
            for e_comp in extracted_components:
                similarity = difflib.SequenceMatcher(None, comp, e_comp).ratio() * 100
                if similarity > best_score:
                    best_score = similarity
                    best_match = e_comp
            component_scores.append(best_score)
            if best_score > 70:
                matching_components.append(f"{comp}≈{best_match}")
    
    # Calculate overall component match score
    if not component_scores:
        component_avg = 0
    else:
        component_avg = sum(component_scores) / len(component_scores)
    
    # Final score (weighted combination of component match and pincode match)
    component_weight = 0.7
    pincode_weight = 0.3
    final_score = (component_avg * component_weight) + (pincode_score * pincode_weight)
    
    # Prepare match details
    match_details = {
        "pincode_match": pincode_match,
        "component_match": component_avg,
        "matching_components": matching_components
    }
    
    # Return final score and match details
    return final_score, match_details

def create_output_table(input_data, extracted_data):
    """
    Generate a comprehensive output table for verification results
    
    Args:
    - input_data (DataFrame): Excel input data with reference information
    - extracted_data (list): List of dictionaries with extracted document information
    
    Returns:
    - list of dictionaries with verification results
    """
    output = []
    
    # Handle case where 'Image No.' might be missing or have different column name
    image_col = 'Image No.' if 'Image No.' in input_data.columns else \
               [col for col in input_data.columns if 'image' in col.lower()]
    
    if not image_col:
        # If no image column found, generate default image numbers
        input_data['Image No.'] = [f'SR{i+1}.jpg' for i in range(len(input_data))]
        image_col = 'Image No.'
    elif isinstance(image_col, list):
        image_col = image_col[0]
    
    # Ensure we have a predictable order of processing
    try:
        sorted_input = input_data.sort_values(by=image_col, 
                                              key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int), 
                                              na_position='first')
    except Exception:
        # Fallback to original order if sorting fails
        sorted_input = input_data.copy()
    
    for idx, row in sorted_input.iterrows():
        # Default image number from input data or generate one
        image_no = row.get(image_col, f'SR{idx+1}.jpg')
        
        # Try to find matching extracted data
        matching_doc = next((doc for doc in extracted_data if doc.get('Image', '') == image_no), None)
        
        # Default values
        doc_type = 'Unknown'
        final_remark = 'Unprocessed'
        accepted = False
        
        if matching_doc:
            # Get document type and standardize it
            doc_type = standardize_document_type(matching_doc.get('Document Type', 'Unknown'))
            
            # Set verification status based on document type and match results
            if doc_type == 'Aadhaar':
                # Get match results
                uid_match = matching_doc.get('UID Match', False)
                name_match_result = matching_doc.get('Name Match', False)
                address_match_result = matching_doc.get('Address Match', False)
                
                # Determine acceptance status and final remarks
                if uid_match and name_match_result and address_match_result:
                    accepted = True
                    final_remark = 'All matched'
                elif not uid_match:
                    final_remark = 'UID mismatch'
                elif not name_match_result:
                    final_remark = 'Name mismatch'
                elif not address_match_result:
                    final_remark = 'Address mismatch'
                else:
                    final_remark = 'Multiple mismatches'
            else:
                # For non-Aadhaar documents
                accepted = False
                final_remark = 'Non-Aadhaar'
        else:
            # No matching document found
            doc_type = 'Non-Aadhaar'
            final_remark = 'Document not found'
        
        # Create output row
        output_row = {
            'Sr.No': len(output) + 1,
            'Image No.': image_no,
            'Document Type': doc_type,
            'Accepted/Rejected': 'Accepted' if accepted else 'Rejected',
            'Final Remarks': final_remark
        }
        
        output.append(output_row)
    
    return output

# Add this to your imports section if not already there
import matplotlib.pyplot as plt
import numpy as np

# Replace your existing generate_visualizations function with this updated version
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import logging
import traceback

def generate_visualizations(results, output_dir):
    """Generate comprehensive visualizations for document verification results"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set custom color palette with corrected naming
        colors = {
            'black': '#000000',
            'white': 'white',
            'gold': '#FFD700',
            'light_purple': '#9999ff',  # Changed from 'dark_blue'
            'purple': '#645cdd',        # Changed from 'green'
            'pale_purple': '#9798de'    # Changed from 'red'
        }


        # Calculate results summaries
        total_docs = len(results)
        accepted = sum(1 for r in results if r['Accepted/Rejected'] == 'Accepted')
        rejected = total_docs - accepted

        # Configure matplotlib
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'axes.labelcolor': colors['light_purple'],
            'axes.titlecolor': colors['light_purple'],
            'text.color': colors['light_purple']
        })

        # 1. Accepted vs Rejected Pie Chart
        plt.figure(figsize=(8, 6))
        pie_labels = ['Accepted', 'Rejected']
        pie_sizes = [accepted, rejected]
        pie_colors = [colors['purple'], colors['pale_purple']]
        
        # Create the pie chart with custom text properties
        patches, texts, autotexts = plt.pie(
                pie_sizes, 
                labels=pie_labels, 
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.1, 0)
        )
        
        # Set outer label text properties (the category names)
        for text in texts:
            text.set_color(colors['light_purple'])
            text.set_fontweight('bold')
            
        # Set inner percentage text properties (make them white)
        for autotext in autotexts:
            autotext.set_color(colors['white'])
            autotext.set_fontweight('bold')
            
        plt.title('Document Verification Breakdown', fontsize=15, fontweight='bold')
        plt.axis('equal')
        
        pie_chart_path = os.path.join(output_dir, 'pie_chart.png')
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight', facecolor=colors['white'])
        plt.close()

        # 2. Final Remarks Bar Chart
        plt.figure(figsize=(10, 6))
        
        # Count occurrences of each final remark
        remarks = {}
        for r in results:
            remark = r.get('Final Remarks', 'Unknown')
            remarks[remark] = remarks.get(remark, 0) + 1
        
        # Sort remarks by frequency
        remarks_sorted = dict(sorted(remarks.items(), key=lambda x: x[1], reverse=True))
        
        # Prepare bar chart data
        bar_labels = list(remarks_sorted.keys())
        bar_heights = list(remarks_sorted.values())
        
        # Create color mapping
        bar_colors = [colors['purple'] if 'Accepted' in label or 'matched' in label.lower() 
                      else colors['pale_purple'] if 'Rejected' in label or 'mismatch' in label.lower() 
                      else colors['light_purple'] for label in bar_labels]
        
        plt.bar(bar_labels, bar_heights, color=bar_colors)
        plt.title('Distribution of Final Remarks', fontsize=15, fontweight='bold')
        plt.xlabel('Remarks', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # Adjust bottom margin to prevent cutting off x-axis labels
        plt.tight_layout()
        
        # Add value labels inside the bars
        for i, v in enumerate(bar_heights):
            plt.text(i, v/2, str(v), 
                     ha='center', va='center', 
                     color=colors['white'], 
                     fontweight='bold')
        
        remarks_chart_path = os.path.join(output_dir, 'remarks_chart.png')
        plt.savefig(remarks_chart_path, dpi=300, bbox_inches='tight', facecolor=colors['white'])
        plt.close()

        # 3. UID Match Score Distribution
        plt.figure(figsize=(10, 6))
        
        # Get UID match scores
        uid_scores = [r.get('UID Match Score', 0) for r in results if 'UID Match Score' in r]
        
        # Create score buckets
        buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bucket_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
        
        # Count frequencies
        uid_hist, _ = np.histogram(uid_scores, bins=buckets)
        
        # Color mapping for bars (pale_purple for below 70, purple for 70 and above)
        uid_bar_colors = [colors['pale_purple'] if i < 7 else colors['purple'] for i in range(len(bucket_labels))]
        
        plt.bar(bucket_labels, uid_hist, color=uid_bar_colors)
        plt.title('UID Match Score Distribution', fontsize=15, fontweight='bold')
        plt.xlabel('Match Score (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels inside the bars
        for i, v in enumerate(uid_hist):
            if v > 0:
                plt.text(i, v/2, str(v), 
                         ha='center', va='center', 
                         color=colors['white'], 
                         fontweight='bold')
        
        plt.tight_layout()
        uid_chart_path = os.path.join(output_dir, 'uid_score_chart.png')
        plt.savefig(uid_chart_path, dpi=300, bbox_inches='tight', facecolor=colors['white'])
        plt.close()

        # 4. Address Match Score Distribution
        plt.figure(figsize=(10, 6))
        
        # Get Address match scores
        address_scores = [r.get('Address Match Score', 0) for r in results if 'Address Match Score' in r]
        
        # Count frequencies
        address_hist, _ = np.histogram(address_scores, bins=buckets)
        
        # Color mapping for bars (pale_purple for below 70, purple for 70 and above)
        address_bar_colors = [colors['pale_purple'] if i < 7 else colors['purple'] for i in range(len(bucket_labels))]
        
        plt.bar(bucket_labels, address_hist, color=address_bar_colors)
        plt.title('Address Match Score Distribution', fontsize=15, fontweight='bold')
        plt.xlabel('Match Score (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels inside the bars
        for i, v in enumerate(address_hist):
            if v > 0:
                plt.text(i, v/2, str(v), 
                         ha='center', va='center', 
                         color=colors['white'], 
                         fontweight='bold')
        
        plt.tight_layout()
        address_chart_path = os.path.join(output_dir, 'address_score_chart.png')
        plt.savefig(address_chart_path, dpi=300, bbox_inches='tight', facecolor=colors['white'])
        plt.close()

        # Return chart URLs (ensure these match your Flask route)
        return {
            'pie_chart': '/static/pie_chart.png',
            'remarks_chart': '/static/remarks_chart.png',
            'uid_score_chart': '/static/uid_score_chart.png',
            'address_score_chart': '/static/address_score_chart.png'
        }
        
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        traceback.print_exc()
        return {}
    
@app.route('/history')
def history():
    """Display verification history"""
    try:
        if DB_CONNECTION and hasattr(DB_CONNECTION, 'is_connected') and DB_CONNECTION.is_connected():
            # Fetch session history from database
            cursor = DB_CONNECTION.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM processing_sessions 
                ORDER BY created_at DESC
            """)
            sessions = cursor.fetchall()
            cursor.close()
            
            # Convert date strings to Python datetime objects if needed
            for session in sessions:
                if isinstance(session['created_at'], str):
                    session['created_at'] = datetime.strptime(session['created_at'], '%Y-%m-%d %H:%M:%S')
            
            return render_template('history.html', sessions=sessions)
        else:
            flash("Database connection not available", "warning")
            return render_template('history.html', sessions=[])
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        flash(f"An error occurred while fetching history: {str(e)}", "error")
        return render_template('history.html', sessions=[])

@app.route('/view-session/<session_id>')
def view_session(session_id):
    """View details of a specific verification session"""
    try:
        if not DB_CONNECTION or not hasattr(DB_CONNECTION, 'is_connected') or not DB_CONNECTION.is_connected():
            flash("Database connection not available", "warning")
            return redirect(url_for('history'))
            
        # Fetch session information
        cursor = DB_CONNECTION.cursor(dictionary=True)
        
        # Get session details
        cursor.execute("""
            SELECT * FROM processing_sessions 
            WHERE session_id = %s
        """, (session_id,))
        session_info = cursor.fetchone()
        
        if not session_info:
            flash("Session not found", "error")
            return redirect(url_for('history'))
            
        # Convert date string to Python datetime object if needed
        if isinstance(session_info['created_at'], str):
            session_info['created_at'] = datetime.strptime(session_info['created_at'], '%Y-%m-%d %H:%M:%S')
        
        # Get verification results
        cursor.execute("""
            SELECT vr.*, ed.extracted_uid, ed.extracted_name, ed.extracted_address, 
                   ed.input_uid, ed.input_name, ed.input_address
            FROM verification_results vr
            LEFT JOIN extracted_data ed ON vr.id = ed.result_id
            WHERE vr.session_id = %s
            ORDER BY vr.id
        """, (session_id,))
        results = cursor.fetchall()
        cursor.close()
        
        return render_template('session.html', session_info=session_info, results=results)
        
    except Exception as e:
        logger.error(f"Error viewing session: {str(e)}")
        flash(f"An error occurred while viewing session: {str(e)}", "error")
        return redirect(url_for('history'))
       
def cleanup_old_sessions():
    """Background job to clean up old session directories"""
    try:
        for session_dir in os.listdir(UPLOAD_DIR):
            session_path = os.path.join(UPLOAD_DIR, session_dir)
            if os.path.isdir(session_path):
                # Check if directory is older than 24 hours
                mtime = os.path.getmtime(session_path)
                age_hours = (time.time() - mtime) / 3600
                
                if age_hours > 24:
                    shutil.rmtree(session_path)
                    logger.info(f"Cleaned up old session directory: {session_path}")
    except Exception as e:
        logger.error(f"Error in cleanup job: {str(e)}")

# Routes
@app.route('/')
@app.route('/home')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/services')
def services():
    """Render the services/upload page"""
    return render_template('services.html')

@app.route('/contact')
def contact():
    """Render the contact page"""
    return render_template('contact.html')

@app.route('/results')
def results():
    """Render the results page with verification results and visualizations"""
    if 'results_data' not in session:
        flash("No verification results available. Please upload files first.", "error")
        return redirect(url_for('services'))
    
    # Get results data from session
    results_data = session.get('results_data', {})
    
    return render_template(
        'results.html',
        results=results_data.get('results', []),
        summary=results_data.get('summary', {}),
        charts=results_data.get('charts', {})
    )
def process_excel_with_verification(excel_file_path, output_dir):
    """
    Process an existing Excel file with verification data and update it with results
    
    Args:
        excel_file_path: Path to the Excel file with verification data
        output_dir: Directory to save output files
        
    Returns:
        Updated DataFrame with verification results
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path)
        
        # Create output rows list for verification results
        verification_results = []
        
        # Process each row in the Excel file
        for idx, row in df.iterrows():
            # Extract necessary information
            input_name = row.get('Name', '')
            input_uid = str(row.get('UID', ''))
            extracted_name = row.get('Extracted Name', '')
            extracted_uid = str(row.get('Extracted UID', ''))
            
            # Reconstruct addresses from components if available
            input_address_parts = []
            
            # Check for address components in specific order
            if pd.notna(row.get('Floor Number', '')):
                input_address_parts.append(str(row.get('Floor Number', '')))
            
            if pd.notna(row.get('Premise Building Name', '')):
                input_address_parts.append(str(row.get('Premise Building Name', '')))
                
            if pd.notna(row.get('Street Road Name', '')):
                input_address_parts.append(str(row.get('Street Road Name', '')))
                
            if pd.notna(row.get('Landmark', '')):
                input_address_parts.append(str(row.get('Landmark', '')))
                
            if pd.notna(row.get('City', '')):
                input_address_parts.append(str(row.get('City', '')))
                
            if pd.notna(row.get('State', '')):
                input_address_parts.append(str(row.get('State', '')))
                
            if pd.notna(row.get('PINCODE', '')):
                input_address_parts.append(str(row.get('PINCODE', '')))
                
            # Join address parts
            input_address = ', '.join(part for part in input_address_parts if part.strip())
            
            # Use the extracted address if available, otherwise use empty string
            extracted_address = row.get('Extracted Address', '')
            
            # Determine document type
            doc_type = row.get('Document Type', 'Unknown')
            doc_type = standardize_document_type(doc_type)
            
            # Perform verification checks
            uid_match = False
            if input_uid and extracted_uid:
                # Clean UIDs for comparison
                clean_input_uid = re.sub(r'\D', '', str(input_uid))
                clean_extracted_uid = re.sub(r'\D', '', str(extracted_uid))
                uid_match = clean_input_uid == clean_extracted_uid
            
            # Calculate match scores
            uid_match_score = calculate_uid_match_score(extracted_uid, input_uid)
            name_match_score, name_match_type = calculate_name_match_score(input_name, extracted_name)
            address_match_score, address_match_details = calculate_address_match_score(input_address, extracted_address)
            
            # Check name match
            name_match_result = name_match(input_name, extracted_name)
            
            # Check address match
            address_match_result = address_match(input_address, extracted_address)
            
            # Determine verification status and remarks
            accepted = False
            if doc_type == 'Aadhaar':
                if uid_match and name_match_result and address_match_result:
                    accepted = True
                    final_remarks = 'All matched'
                elif not uid_match:
                    final_remarks = 'UID mismatch'
                elif not name_match_result:
                    final_remarks = 'Name mismatch'
                elif not address_match_result:
                    final_remarks = 'Address mismatch'
                else:
                    final_remarks = 'Multiple mismatches'
            else:
                final_remarks = 'Non-Aadhaar'
            
            # Update row with verification results
            df.at[idx, 'Name Match Score'] = name_match_score
            df.at[idx, 'UID Match Score'] = uid_match_score
            df.at[idx, 'Final Address Match Score'] = address_match_score
            df.at[idx, 'Final Remarks'] = final_remarks
            df.at[idx, 'Overall Match'] = 'Yes' if accepted else 'No'
            
            # Create a result row for verification summary
            verification_row = {
                'Sr.No': idx + 1,
                'Image No.': f"SR{idx+1}.jpg",
                'Document Type': doc_type,
                'Accepted/Rejected': 'Accepted' if accepted else 'Rejected',
                'Final Remarks': final_remarks,
                'UID Match Score': uid_match_score,
                'Name Match Score': name_match_score,
                'Name Match Type': name_match_type,
                'Address Match Score': address_match_score,
                'Input UID': input_uid,
                'Input Name': input_name,
                'Input Address': input_address,
                'Extracted UID': extracted_uid,
                'Extracted Name': extracted_name,
                'Extracted Address': extracted_address
            }
            
            verification_results.append(verification_row)
        
        # Save updated Excel file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        updated_excel_path = os.path.join(output_dir, f"updated_verification_{timestamp}.xlsx")
        df.to_excel(updated_excel_path, index=False)
        
        # Generate a separate verification results Excel file
        verification_df = pd.DataFrame(verification_results)
        verification_excel_path = os.path.join(output_dir, f"verification_results_{timestamp}.xlsx")
        verification_df.to_excel(verification_excel_path, index=False)
        
        return df, verification_results
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        return None, []
def update_original_excel(excel_path, verification_results):
    """
    Update the original Excel file with verification results
    
    Args:
        excel_path: Path to the original Excel file
        verification_results: Verification results from processing
        
    Returns:
        Path to the updated Excel file
    """
    try:
        # Read the original Excel
        df = pd.read_excel(excel_path)
        
        # Identify the image/sr column
        image_col = 'SrNo' if 'SrNo' in df.columns else (
                    'Image No.' if 'Image No.' in df.columns else 
                    [col for col in df.columns if any(img in col.lower() for img in ['image', 'sr', 'srno', 'serial'])])
        
        if isinstance(image_col, list) and image_col:
            image_col = image_col[0]
        elif not image_col or (isinstance(image_col, list) and not image_col):
            # If no suitable column found, add one
            df['SrNo'] = [f'SR{i+1}' for i in range(len(df))]
            image_col = 'SrNo'
        
        # Map verification results to original Excel format
        for idx, row in df.iterrows():
            # Get the SrNo or image number
            sr_no = str(row.get(image_col, f'{idx+1}'))
            # Ensure SR prefix for consistency if not already there
            if not sr_no.upper().startswith('SR'):
                sr_no = f'SR{sr_no}'
                
            # Find matching verification result by SrNo
            result = next((r for r in verification_results if 
                        r.get('Image No.', '').startswith(sr_no) or 
                        r.get('Image No.', '') == f"{sr_no}.jpg"), None)
            
            if result:
                # Update each column in the original Excel if it exists
                columns_to_update = {
                    'Extracted Name': result.get('Extracted Name', ''),
                    'Extracted UID': result.get('Extracted UID', ''),
                    'Extracted Address': result.get('Extracted Address', ''),
                    'Name match percentage': result.get('Name Match Score', 0),
                    'Name Match Score': result.get('Name Match Score', 0),
                    'UID Match Score': result.get('UID Match Score', 0),
                    'Final Address Match Score': result.get('Address Match Score', 0),
                    'Final Remarks': result.get('Final Remarks', ''),
                    'Document Type': result.get('Document Type', ''),
                    'Overall Match': 'Yes' if result.get('Accepted/Rejected') == 'Accepted' else 'No'
                }
                
                # Update each column if it exists
                for col, value in columns_to_update.items():
                    if col in df.columns:
                        df.at[idx, col] = value
        
        # Create a backup copy with timestamp
        dirname, filename = os.path.split(excel_path)
        name, ext = os.path.splitext(filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(dirname, f"{name}_backup_{timestamp}{ext}")
        df.to_excel(backup_path, index=False)
        
        # Save back to the same file
        df.to_excel(excel_path, index=False)
        
        return excel_path
    
    except Exception as e:
        logger.error(f"Error updating Excel file: {str(e)}")
        traceback.print_exc()  # Print the full stack trace for debugging
        return None
    
@app.route('/process-excel', methods=['POST'])
def process_excel():
    """Process an uploaded Excel file with verification data"""
    try:
        if 'excelFile' not in request.files:
            flash("Please upload an Excel file", "error")
            return redirect(url_for('services'))
            
        excel_file = request.files['excelFile']
        
        if not excel_file.filename.lower().endswith(('.xlsx', '.xls')):
            flash("Please upload a valid Excel file (.xlsx or .xls)", "error")
            return redirect(url_for('services'))
        
        # Create a session directory
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save the Excel file
        excel_path = os.path.join(session_dir, 'verification_data.xlsx')
        excel_file.save(excel_path)
        
        # Process the Excel file
        df, verification_results = process_excel_with_verification(excel_path, STATIC_DIR)
        
        if df is None:
            flash("Error processing the Excel file", "error")
            return redirect(url_for('services'))
        
        # Calculate summary
        summary = {
            'total': len(verification_results),
            'accepted': sum(1 for r in verification_results if r['Accepted/Rejected'] == 'Accepted'),
            'rejected': sum(1 for r in verification_results if r['Accepted/Rejected'] == 'Rejected')
        }
        
        # Generate visualizations
        charts = generate_visualizations(verification_results, STATIC_DIR)
        
        # Store results in session
        session['results_data'] = {
            'results': verification_results,
            'summary': summary,
            'charts': charts
        }
        
        # Save to database if available
        if DB_CONNECTION and hasattr(DB_CONNECTION, 'is_connected') and DB_CONNECTION.is_connected():
            try:
                db_success = db.save_verification_results(DB_CONNECTION, verification_results, session_id)
                if db_success:
                    logger.info(f"Results saved to database for session {session_id}")
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}")
        
        return render_template(
            'results.html',
            results=verification_results,
            summary=summary,
            charts=charts
        )
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for('services'))

@app.route('/process-files', methods=['POST'])
def process_files():
    """Process uploaded files (ZIP of images and Excel data) and generate verification results"""
    try:
        # Validate and save files
        if 'zipFile' not in request.files or 'excelFile' not in request.files:
            flash("Please upload both ZIP and Excel files", "error")
            return redirect(url_for('services'))
            
        zip_file = request.files['zipFile']
        excel_file = request.files['excelFile']
        
        # Validate file types
        if not zip_file.filename.lower().endswith('.zip'):
            flash("Please upload a valid ZIP file", "error")
            return redirect(url_for('services'))
        
        if not excel_file.filename.lower().endswith(('.xlsx', '.xls')):
            flash("Please upload a valid Excel file (.xlsx or .xls)", "error")
            return redirect(url_for('services'))
        
        # Check database connection
        global DB_CONNECTION
        if not DB_CONNECTION or not hasattr(DB_CONNECTION, 'is_connected') or not DB_CONNECTION.is_connected():
            # Try to reconnect
            try:
                DB_CONNECTION = db.create_connection()
                if DB_CONNECTION:
                    logger.info("Database connection re-established")
            except Exception as e:
                logger.error(f"Database reconnection failed: {str(e)}")
        
        # Create a unique session directory
        session_id = str(uuid.uuid4())
        # Store session_id in the session for later use
        session['session_id'] = session_id
        
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save files
        zip_path = os.path.join(session_dir, 'documents.zip')
        excel_path = os.path.join(session_dir, 'data.xlsx')
        zip_file.save(zip_path)
        excel_file.save(excel_path)
        
        # Load Excel file
        try:
            engine = 'openpyxl' if excel_file.filename.endswith('.xlsx') else 'xlrd'
            df = pd.read_excel(excel_path, engine=engine)
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            flash(f"Error reading Excel file: {str(e)}", "error")
            return redirect(url_for('services'))
        
        # Extract ZIP file
        extract_dir = os.path.join(session_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {str(e)}")
            flash(f"Error extracting ZIP file: {str(e)}", "error")
            return redirect(url_for('services'))
        
        # Ensure 'Image No.' column exists
        image_col = 'SrNo' if 'SrNo' in df.columns else (
                  'Image No.' if 'Image No.' in df.columns else 
                  [col for col in df.columns if any(img in col.lower() for img in ['image', 'sr', 'srno', 'serial'])])
        
        if not image_col or (isinstance(image_col, list) and not image_col):
            df['SrNo'] = [f'SR{i+1}' for i in range(len(df))]
            image_col = 'SrNo'
        elif isinstance(image_col, list):
            image_col = image_col[0]
        
        # Process each document - modified to handle multiple images for the same document
        extracted_results = []
        
        for idx, row in df.iterrows():
            # Get base image filename without the suffix
            base_sr = str(row.get(image_col, f'{idx+1}'))
            # Ensure SR prefix for consistency
            if not base_sr.upper().startswith('SR'):
                base_sr = f'SR{base_sr}'
                
            base_image_no = f"{base_sr}.jpg"
            base_image_id = base_sr
            
            # Look for all images that match this base image ID
            matching_images = []
            for filename in os.listdir(extract_dir):
                if filename.startswith(base_image_id) and is_valid_image(os.path.join(extract_dir, filename)):
                    matching_images.append(filename)
                elif filename == base_image_no and is_valid_image(os.path.join(extract_dir, filename)):
                    matching_images.append(filename)
            
            # Skip if no matching images found
            if not matching_images:
                logger.warning(f"No images found for {base_image_no}")
                continue
            
            # Process each matching image and combine the results
            combined_extracted_data = {
                'uid': None,
                'name': None,
                'address': None
            }
            doc_types = []
            
            for img_name in matching_images:
                image_path = os.path.join(extract_dir, img_name)
                extracted_data, doc_type = extract_text_with_detection(image_path, MODELS)
                
                # Collect document types
                doc_types.append(doc_type)
                
                # Combine extracted data - take the first non-None value for each field
                if extracted_data['uid'] and not combined_extracted_data['uid']:
                    combined_extracted_data['uid'] = extracted_data['uid']
                if extracted_data['name'] and not combined_extracted_data['name']:
                    combined_extracted_data['name'] = extracted_data['name']
                if extracted_data['address'] and not combined_extracted_data['address']:
                    combined_extracted_data['address'] = extracted_data['address']
            
            # Determine the most likely document type with proper formatting
            if 'Aadhaar' in doc_types or any(d.lower().replace('_', '') in ['aadhar', 'aadhaar'] for d in doc_types):
                final_doc_type = 'Aadhaar'
            elif any(d.lower().replace('_', ' ') in ['non aadhar', 'non aadhaar', 'non_aadhar'] for d in doc_types):
                final_doc_type = 'Non-Aadhaar'
            else:
                final_doc_type = doc_types[0] if doc_types else "Unknown"
                # Apply standardization to ensure proper formatting
                final_doc_type = standardize_document_type(final_doc_type)
            
            # Construct input reference data
            input_uid = str(row.get('UID', '')) if pd.notna(row.get('UID', '')) else ''
            input_name = str(row.get('Name', '')) if pd.notna(row.get('Name', '')) else ''
            
            # Construct input address from components or use Address field if available
            if 'Address' in row and pd.notna(row['Address']):
                input_address = str(row['Address'])
            else:
                input_address = reconstruct_address(row)
            
            # Perform matching
            uid_match = False
            name_match_result = False
            address_match_result = False
            
            # UID matching
            if combined_extracted_data['uid'] and input_uid:
                uid_match = str(combined_extracted_data['uid']) == str(input_uid)
            
            # Name matching
            if combined_extracted_data['name'] and input_name:
                name_match_result = name_match(input_name, combined_extracted_data['name'])
            
            # Address matching
            if combined_extracted_data['address'] and input_address:
                address_match_result = address_match(input_address, combined_extracted_data['address'])
            
            # Calculate match scores
            uid_match_score = calculate_uid_match_score(combined_extracted_data['uid'], input_uid)
            name_match_score, name_match_type = calculate_name_match_score(input_name, combined_extracted_data['name'])
            address_match_score, address_match_details = calculate_address_match_score(input_address, combined_extracted_data['address'])
            
            # Store results
            result = {
                'Image': base_image_no,  # Use the original image number from the Excel file
                'Document Type': final_doc_type,
                'Extracted UID': combined_extracted_data['uid'],
                'Extracted Name': combined_extracted_data['name'],
                'Extracted Address': combined_extracted_data['address'],
                'Input UID': input_uid,
                'Input Name': input_name,
                'Input Address': input_address,
                'UID Match': uid_match,
                'Name Match': name_match_result,
                'Address Match': address_match_result,
                'UID Match Score': uid_match_score,
                'Name Match Score': name_match_score,
                'Name Match Type': name_match_type,
                'Address Match Score': address_match_score,
                'Address Match Details': address_match_details
            }
            
            extracted_results.append(result)
        
        # Create output table with updated logic for acceptance criteria
        output = []
        
        # Ensure we have a predictable order of processing
        try:
            sorted_input = df.sort_values(by=image_col)
        except Exception:
            # Fallback to original order if sorting fails
            sorted_input = df.copy()
        
        for idx, row in sorted_input.iterrows():
            # Get the SrNo or image number
            sr_no = str(row.get(image_col, f'{idx+1}'))
            # Ensure SR prefix for consistency
            if not sr_no.upper().startswith('SR'):
                sr_no = f'SR{sr_no}'
                
            image_no = f"{sr_no}.jpg"
            
            # Try to find matching extracted data
            matching_doc = next((doc for doc in extracted_results if 
                               doc.get('Image', '').startswith(sr_no) or 
                               doc.get('Image', '') == image_no), None)
            
            # Default values
            doc_type = 'Unknown'
            final_remark = 'Unprocessed'
            accepted = False
            
            # Initialize match scores and details
            uid_match_score = 0
            name_match_score = 0
            name_match_type = 'No match'
            address_match_score = 0
            address_match_details = {"pincode_match": False, "component_match": 0}
            input_uid = ''
            input_name = ''
            input_address = ''
            extracted_uid = ''
            extracted_name = ''
            extracted_address = ''
            
            if matching_doc:
                # Get document type and standardize it
                doc_type = standardize_document_type(matching_doc.get('Document Type', 'Unknown'))
                
                # Get input and extracted data
                input_uid = matching_doc.get('Input UID', '')
                input_name = matching_doc.get('Input Name', '')
                input_address = matching_doc.get('Input Address', '')
                extracted_uid = matching_doc.get('Extracted UID', '')
                extracted_name = matching_doc.get('Extracted Name', '')
                extracted_address = matching_doc.get('Extracted Address', '')
                
                # Get match scores
                uid_match_score = matching_doc.get('UID Match Score', 0)
                name_match_score = matching_doc.get('Name Match Score', 0)
                name_match_type = matching_doc.get('Name Match Type', 'No match')
                address_match_score = matching_doc.get('Address Match Score', 0)
                address_match_details = matching_doc.get('Address Match Details', {"pincode_match": False, "component_match": 0})
                
                # Set verification status based on document type and match results
                if doc_type == 'Aadhaar':
                    # Get match results
                    uid_match = matching_doc.get('UID Match', False)
                    name_match_result = matching_doc.get('Name Match', False)
                    address_match_result = matching_doc.get('Address Match', False)
                    
                    # Determine acceptance status and final remarks
                    if uid_match and name_match_result and address_match_result:
                        accepted = True
                        final_remark = 'All matched'
                    elif not uid_match:
                        final_remark = 'UID mismatch'
                    elif not name_match_result:
                        final_remark = 'Name mismatch'
                    elif not address_match_result:
                        final_remark = 'Address mismatch'
                    else:
                        final_remark = 'Multiple mismatches'
                else:
                    # For non-Aadhaar documents
                    accepted = False
                    final_remark = 'Non-Aadhaar'
            else:
                # No matching document found
                doc_type = 'Non-Aadhaar'
                final_remark = 'Document not found'
            
            # Standardize document type format for consistent display
            doc_type = standardize_document_type(doc_type)
            
            # Create output row
            output_row = {
                'Sr.No': len(output) + 1,
                'Image No.': image_no,
                'Document Type': doc_type,
                'Accepted/Rejected': 'Accepted' if accepted else 'Rejected',
                'Final Remarks': final_remark,
                'UID Match Score': uid_match_score,
                'Name Match Score': name_match_score,
                'Name Match Type': name_match_type,
                'Address Match Score': address_match_score,
                'Address Match Details': address_match_details,
                'Input UID': input_uid,
                'Input Name': input_name,
                'Input Address': input_address,
                'Extracted UID': extracted_uid,
                'Extracted Name': extracted_name,
                'Extracted Address': extracted_address
            }
            
            output.append(output_row)
        
        # Calculate summary
        summary = {
            'total': len(output),
            'accepted': sum(1 for r in output if r['Accepted/Rejected'] == 'Accepted'),
            'rejected': sum(1 for r in output if r['Accepted/Rejected'] == 'Rejected')
        }
        
        # Generate visualizations
        charts = generate_visualizations(output, STATIC_DIR)
        
        # Store results in session
        session['results_data'] = {
            'results': output,
            'summary': summary,
            'charts': charts
        }
        
        # Update the original Excel file with verification results
        original_excel_updated = False
        if extracted_results:
            updated_excel_path = update_original_excel(excel_path, output)
            if updated_excel_path:
                logger.info(f"Original Excel file updated at {updated_excel_path}")
                # Store path for download
                session['excel_path'] = updated_excel_path
                original_excel_updated = True
            else:
                logger.warning("Failed to update original Excel file")

        # Only create standard Excel file if original update failed
        if not original_excel_updated:
            excel_df = pd.DataFrame(output)
            fallback_path = os.path.join(STATIC_DIR, 'excel_outputs', f"verification_report_{session_id}.xlsx")
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            excel_df.to_excel(fallback_path, index=False)
            session['excel_path'] = fallback_path
            logger.info(f"Created new Excel file at {fallback_path} as fallback")
        
        # Save results to database if connection is available
        if DB_CONNECTION and hasattr(DB_CONNECTION, 'is_connected') and DB_CONNECTION.is_connected():
            try:
                db_success = db.save_verification_results(DB_CONNECTION, output, session_id)
                if db_success:
                    logger.info(f"Results saved to database successfully for session {session_id}")
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}")
        
        # CHANGE HERE: Redirect to results page instead of rendering in services.html
        return redirect(url_for('results'))
        
    except Exception as e:
        # Add error handling for any exceptions that might occur
        logger.error(f"Error processing files: {str(e)}")
        flash(f"An error occurred while processing your files: {str(e)}", "error")
        return redirect(url_for('services'))
    
# Update the download-report route in app.py to use the saved Excel file:

@app.route('/download-report')
def download_report():
    """Generate and download a verification report"""
    if 'results_data' not in session:
        flash("No verification results available. Please upload files first.", "error")
        return redirect(url_for('services'))
    
    # Get results data from session
    results_data = session.get('results_data', {})
    results = results_data.get('results', [])
    summary = results_data.get('summary', {})
    
    try:
        # Check if user wants Excel or text report (default to Excel)
        report_type = request.args.get('type', 'excel')
        
        if report_type == 'excel':
            # First check for the updated original Excel file
            if 'excel_path' in session and os.path.exists(session['excel_path']):
                excel_path = session['excel_path']
                
                # Get filename from path
                excel_filename = os.path.basename(excel_path)
                
                # Return the updated Excel file
                with open(excel_path, 'rb') as f:
                    buffer = BytesIO(f.read())
                
                return Response(
                    buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={
                        'Content-Disposition': f'attachment; filename="{excel_filename}"'
                    }
                )
            # If no saved Excel file or DB connection, generate on-the-fly
            elif DB_CONNECTION and DB_CONNECTION.is_connected() and 'session_id' in session:
                # Export to Excel using database data
                df = db.export_to_excel(DB_CONNECTION, session['session_id'])
                if df is not None:
                    buffer = BytesIO()
                    df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    
                    return Response(
                        buffer.getvalue(),
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        headers={
                            'Content-Disposition': 'attachment; filename=verification_report.xlsx'
                        }
                    )
            else:
                # Generate Excel directly from session data
                df = pd.DataFrame(results)
                buffer = BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                
                return Response(
                    buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={
                        'Content-Disposition': 'attachment; filename=verification_report.xlsx'
                    }
                )
            
        # Generate text report if Excel not available or requested
        if report_type == 'text':
            buffer = BytesIO()
            
            # Create a simple text-based report
            report_text = f"""
            Aadhaar Verification Report
            --------------------------
            
            Summary:
            Total Documents: {summary.get('total', 0)}
            Accepted Documents: {summary.get('accepted', 0)}
            Rejected Documents: {summary.get('rejected', 0)}
            
            Detailed Results:
            """
            
            for result in results:
                report_text += f"""
                Document: {result.get('Image No.', 'Unknown')}
                Type: {result.get('Document Type', 'Unknown')}
                Status: {result.get('Accepted/Rejected', 'Unknown')}
                Remarks: {result.get('Final Remarks', 'None')}
                --------------------------
                """
            
            # Write the text to the buffer
            buffer.write(report_text.encode('utf-8'))
            buffer.seek(0)
            
            # Return the buffer as a downloadable file
            return Response(
                buffer,
                mimetype='text/plain',
                headers={
                    'Content-Disposition': 'attachment; filename=verification_report.txt'
                }
            )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        flash(f"Error generating report: {str(e)}", "error")
        return redirect(url_for('results'))@app.teardown_appcontext
def close_db_connection(exception):
    """Close database connection when app context ends"""
    global DB_CONNECTION
    if DB_CONNECTION and hasattr(DB_CONNECTION, 'is_connected') and DB_CONNECTION.is_connected():
        db.close_connection(DB_CONNECTION)
    
# You might also want to add this at the end if it's not there
if __name__ == '__main__':
    # Start a background thread for cleanup
    cleanup_thread = threading.Thread(target=cleanup_old_sessions)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)