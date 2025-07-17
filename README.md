# AadharVerify. - An AI Based Fraud Management System For UID Aadhar

AadharVerify. is an AI-powered document verification platform designed specifically for Aadhaar identity documents. The system uses machine learning models for automated document classification, data extraction, and verification to prevent fraud and ensure regulatory compliance.

The below are the snapshots of the website containing the glimpse of home page and the results:

![Image](https://github.com/user-attachments/assets/bdb5e7d9-1cb5-4e1d-8a4f-56cc0e90a2a2)

![Image](https://github.com/user-attachments/assets/6ede186a-9f48-4743-95b3-31676f600209)

![Image](https://github.com/user-attachments/assets/76f9fb15-25bc-4176-8330-38c8a63cb444)


# Features
•	AI-Powered Document Recognition: Automatically identify and classify Aadhaar and non-Aadhaar documents using YOLO-based machine learning models.

•	Automated Data Extraction: Extract key information such as UID number, name, and address from documents using EasyOCR and custom detection models.

•	Intelligent Matching Algorithms: Compare extracted data with reference information using advanced fuzzy matching and similarity scoring algorithms.

•	Bulk Processing: Process multiple documents simultaneously for enterprise-scale operations with ZIP file support.

•	Comprehensive Reporting: Generate detailed verification reports, match scores, and data visualizations.

•	Verification History: Track and review all verification activities with detailed session information.

•	Interactive Dashboard: Visualize verification metrics and track performance over time.

•	Database Integration: Store verification results for long-term record keeping (optional).

•	Excel Report Generation: Export verification results in structured Excel format for further analysis.

# Detailed File Descriptions
Core Files:

•	app.py: The main Flask application that handles routing, request processing, and integrates all components. Contains the document verification logic, image processing, and result generation functionality.

•	mysql_connection.py: Manages database connections and operations. Creates database schema, handles verification result storage, and provides export functionality.

•	dashboard.py: Implements the analytics dashboard and metrics visualization. Defines routes for dashboard views and generates visualization data.

•	auto_save_excel.py: Handles Excel report generation and formatting. Contains functions to create detailed verification reports with match scores and result details.

•	standalone_script.py: Command-line utility for testing document extraction without using the web interface. Useful for individual document verification and debug purposes.

HTML Templates:

•	base.html: Base template that defines common layout elements including header, footer, and navigation. Other templates extend this file.

•	index.html: Main landing page template with feature description and service introduction.

•	services.html: File upload interface for document verification with drag-and-drop functionality and upload progress indicators.

•	results.html: Displays verification results with tabular data and visualization charts. Includes tab interface for different result views.

•	history.html: Shows past verification sessions with summary statistics. Allows navigation to individual session details.

•	session.html: Displays detailed information about a specific verification session with match scores and verification status.

•	about.html, contact.html: Static information pages about the service and contact details.

•	404.html, 500.html: Custom error pages for not found and server error responses.

Machine Learning Models:

•	classification.pt: YOLO-based model for document type classification. Identifies Aadhaar vs. non-Aadhaar documents.

•	detection.pt: YOLO-based model for text field detection. Locates UID, name, and address regions in documents.

Functions and Algorithms:

•	extract_text_with_detection(): Integrates YOLO models with OCR to extract text from specified regions.

•	calculate_uid_match_score(): Computes similarity score for UID numbers with exact and partial matching.

•	calculate_name_match_score(): Implements sophisticated name matching with support for abbreviations and name order variations.

•	calculate_address_match_score(): Performs component-based address matching with pincode validation and fuzzy matching.

•	generate_visualizations(): Creates charts and visualizations for verification results analysis.

•	create_output_table(): Generates comprehensive result tables with verification status and match scores.

•	process_files(): Orchestrates the entire verification workflow from file upload to result generation.



